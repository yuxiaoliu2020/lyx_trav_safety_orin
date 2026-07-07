import os
import sys
import rospy
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import bisect
import threading
import queue
import time
import message_filters
import tf2_ros
import geometry_msgs.msg
from contextlib import nullcontext
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid
from collections import deque
from scipy.spatial.transform import Rotation as R
import statistics
from jtop import jtop

# 导入自定义消息类型 - 需要在CMakeLists.txt和package.xml中正确配置
from trav_safety.msg import TraversabilityMap, SafetyMap

# 导入模型和配置
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.infer.infer_configs import get_cfg
from src.infer.infer_module_safety import InferModuleSafety as InferModule
# from src.infer.infer_module import InferModule


class FrameBuffer:
    """
    滑动窗口缓冲区，用于存储最近的历史帧数据
    """
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.color_imgs = deque(maxlen=buffer_size)
        self.depth_imgs = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.intrinsics = deque(maxlen=buffer_size)
        self.cam2base = deque(maxlen=buffer_size)
        self.positions = deque(maxlen=buffer_size)
        self.quaternions = deque(maxlen=buffer_size)
        
    def add_frame(self, color_img, depth_img, timestamp, intrinsics, cam2base, position, quaternion):
        """添加新帧到缓冲区"""
        self.color_imgs.append(color_img)
        self.depth_imgs.append(depth_img)
        self.timestamps.append(timestamp)
        self.intrinsics.append(intrinsics)
        self.cam2base.append(cam2base)
        self.positions.append(position)
        self.quaternions.append(quaternion)
        
    def is_full(self):
        """检查缓冲区是否已装满所需的帧数"""
        return len(self.color_imgs) == self.buffer_size
        
    def get_frame_data(self):
        """获取缓冲区中所有帧的数据"""
        return {
            'color_imgs': list(self.color_imgs),
            'depth_imgs': list(self.depth_imgs),
            'timestamps': list(self.timestamps),
            'intrinsics': list(self.intrinsics),
            'cam2base': list(self.cam2base),
            'positions': list(self.positions),
            'quaternions': list(self.quaternions)
        }


class TraversabilityInferenceNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('trav_safety_node')
        
        # 从参数服务器加载配置
        config_file = rospy.get_param('~config_file', None)
        self.config = get_cfg(config_file)

        # 添加TF广播器
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # 加载ROS参数
        self.rgb_topic = rospy.get_param('~rgb_topic', '/zed2i/zed_node/left/image_rect_color')
        self.depth_topic = rospy.get_param('~depth_topic', '/zed2i/zed_node/depth/depth_registered')
        self.pose_topic = rospy.get_param('~pose_topic', '/fixposition/odometry_enu')
        self.model_path = rospy.get_param('~model_path', '/home/lyx/lyx_trav_safety/src/trav_safety/src/checkpoints/safety_bev.ckpt')
        self.trav_map_topic = rospy.get_param('~trav_map_topic', '/traversability_map')
        self.trav_viz_mu_topic = rospy.get_param('~trav_viz_mu_topic', '/traversability_map_viz/mu')
        self.trav_viz_nu_topic = rospy.get_param('~trav_viz_nu_topic', '/traversability_map_viz/nu')
        self.safety_map_topic = rospy.get_param('~safety_map_topic', '/safety_map')
        self.safety_viz_topic = rospy.get_param('~safety_viz_topic', '/safety_map_viz')
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 初始化缓冲区
        self.n_frames = self.config.MODEL.TIME_LENGTH  # 通常为6
        self.current_frame = None
        self.frame_buffer = FrameBuffer(buffer_size=self.n_frames-1)  # 保存5帧历史数据
        
        # 设置网格参数
        self.grid_bounds = self.config.MODEL.GRID_BOUNDS
        
        # 创建线程安全队列
        # 仅保留“最新任务 / 最新结果”，避免队列堆积引起秒级延迟。
        self.inference_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        # 创建设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"使用设备: {self.device}")

        self.safety_head = rospy.get_param('~safety_head', True)

        # 是否启用混合精度推理（仅在CUDA可用时生效）
        self.use_amp = rospy.get_param('~use_amp', self.device.type == 'cuda')
        if self.device.type == 'cuda':
            # 启用 cuDNN 内核自动优化（输入尺寸固定时通常能提速）
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            rospy.loginfo("已启用 cuDNN benchmark 优化")

            if self.use_amp:
                rospy.loginfo("启用 CUDA AMP 混合精度推理")
            else:
                rospy.loginfo("未启用混合精度推理（use_amp 参数为 False）")
        else:
            if self.use_amp:
                rospy.logwarn("当前设备不支持CUDA，已忽略 use_amp 参数，使用纯FP32 推理")
                self.use_amp = False
        
        # 初始化模型
        rospy.loginfo("正在加载模型...")
        self.model = InferModule(self.config)
        self.model = self.model.to(self.device)
        
        # 加载预训练模型权重
        if self.model_path is not None:
            self.config.MODEL.LOAD_NETWORK = self.model_path
            
        if self.config.MODEL.LOAD_NETWORK is not None:
            rospy.loginfo(f'从 {self.config.MODEL.LOAD_NETWORK} 加载预训练模型')
            pretrained_dict = torch.load(self.config.MODEL.LOAD_NETWORK, map_location=self.device)['state_dict']
            self.model.load_state_dict(pretrained_dict, strict=False)
        
        # 设置为评估模式
        self.model.eval()
        
        # 创建数据存储变量
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.latest_info_msg = None
        self.latest_pose_msg = None
        self.data_lock = threading.Lock()
        
        # 创建发布者
        self.trav_map_pub = rospy.Publisher(self.trav_map_topic, TraversabilityMap, queue_size=1)
        self.trav_viz_mu_pub = rospy.Publisher(self.trav_viz_mu_topic, OccupancyGrid, queue_size=1)
        self.trav_viz_nu_pub = rospy.Publisher(self.trav_viz_nu_topic, OccupancyGrid, queue_size=1)
        self.safety_map_pub = rospy.Publisher(self.safety_map_topic, SafetyMap, queue_size=1)
        self.safety_viz_pub = rospy.Publisher(self.safety_viz_topic, OccupancyGrid, queue_size=1)
        
        # 创建独立的订阅者，替代消息过滤器
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        rospy.Subscriber(self.pose_topic, Odometry, self.pose_callback, queue_size=1)

        # 添加2.5Hz定时器
        self.timer = rospy.Timer(rospy.Duration(0.4), self.timer_callback)  # 2.5Hz = 0.4秒

        # 启动线程
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.publishing_thread = threading.Thread(target=self.publishing_loop)
        
        self.inference_thread.daemon = True
        self.publishing_thread.daemon = True
        
        self.inference_thread.start()
        self.publishing_thread.start()

        # 添加性能监控变量
        self.processing_times = deque(maxlen=100)  # 存储最近100次处理的时间
        self.preprocessing_times = deque(maxlen=100)  # 预处理耗时
        self.inference_times = deque(maxlen=100)  # 推理耗时
        self.postprocessing_times = deque(maxlen=100)  # 后处理耗时
        self.total_times = deque(maxlen=100)  # 端到端总耗时
        
        self.last_frame_time = None  # 上一帧处理完成的时间
        self.processing_freqs = deque(maxlen=100)  # 处理频率

        self.power_readings = deque(maxlen=100)  # 存储最近100次功耗读数
        self.jetson_available = False

        self.frame_counter = 0
        self.warmup_frames = 20  # 预热帧数

        try:
            self.jetson = jtop()
            self.jetson.start()
            if self.jetson.ok():
                rospy.loginfo("Jetson功耗监控成功初始化")
                # 测试是否能实际获取功耗数据
                power_dict = self.jetson.power
                rospy.loginfo(f"功耗字典内容: {power_dict}")
                if power_dict and 'tot' in power_dict and 'power' in power_dict['tot']:
                    total_power = power_dict['tot']['power']
                    rospy.loginfo(f"检测到当前功耗: {total_power/1000.0:.2f}W")
                else:
                    rospy.logwarn("Jetson功耗数据为空或格式异常，可能需要root权限")
            else:
                rospy.logwarn("Jetson功耗监控初始化失败，无法获取硬件信息")
            self.jetson_available = self.jetson.ok()
        except Exception as e:
            rospy.logerr(f"Jetson功耗监控初始化错误: {e}")
            self.jetson_available = False
            
        # 启动功耗监控线程
        self.power_monitor_thread = threading.Thread(target=self.power_monitor_loop)
        self.power_monitor_thread.daemon = True
        self.power_monitor_thread.start()
        rospy.loginfo("Jetson功耗监控已启动")
        
        # 添加定时器，每10秒输出一次性能统计信息
        rospy.Timer(rospy.Duration(10.0), self.print_performance_stats)

        # 发布静态坐标系变换
        self.publish_static_transforms()
        
        rospy.loginfo("通过性推理节点初始化完成！设置为2.5Hz处理频率")
    
    def __del__(self):
        """析构函数，确保资源正确释放"""
        if hasattr(self, 'jetson') and self.jetson_available:
            try:
                self.jetson.close()
            except:
                pass
    
    def power_monitor_loop(self):
        """定期采样Jetson的功耗数据"""
        sample_count = 0
        
        while not rospy.is_shutdown() and self.jetson_available:
            try:
                # 获取当前功耗数据
                if self.jetson.ok():
                    # 获取总功耗 (单位: mW)
                    power_dict = self.jetson.power
                    
                    # 直接从'tot'字段获取总功耗
                    if power_dict and 'tot' in power_dict and 'power' in power_dict['tot']:
                        total_power = power_dict['tot']['power']
                        
                        if self.frame_counter >= self.warmup_frames:
                            # 存储功耗数据 (转换为W)
                            self.power_readings.append(total_power / 1000.0)
                            sample_count += 1
                            if sample_count % 50 == 0:  # 每5秒报告一次
                                rospy.loginfo(f"功耗监控: 当前功耗={total_power/1000.0:.2f}W")
                
                # 每100ms采样一次
                rospy.sleep(0.1)
            except Exception as e:
                rospy.logerr(f"功耗监控错误: {e}")
                self.jetson_available = False
                break
    
    def print_performance_stats(self, event=None):
        """输出性能统计信息"""
        if len(self.total_times) < 10:
            if self.frame_counter < self.warmup_frames:
                rospy.loginfo(f"正在预热: {self.frame_counter}/{self.warmup_frames} 帧")
            else:
                rospy.loginfo(f"预热完成({self.warmup_frames}帧)，等待收集更多性能数据...")
            return
        
        # 计算平均值和标准差
        avg_preprocess = sum(self.preprocessing_times) / len(self.preprocessing_times)
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        avg_postprocess = sum(self.postprocessing_times) / len(self.postprocessing_times)
        avg_total = sum(self.total_times) / len(self.total_times)
        
        # 计算频率统计
        if len(self.processing_freqs) > 0:
            avg_freq = sum(self.processing_freqs) / len(self.processing_freqs)
            if len(self.processing_freqs) > 1:
                freq_std = statistics.stdev(self.processing_freqs)
            else:
                freq_std = 0.0
        else:
            avg_freq = 0.0
            freq_std = 0.0

        avg_power = sum(self.power_readings) / len(self.power_readings)
        if len(self.power_readings) > 1:
            power_std = statistics.stdev(self.power_readings)
        else:
            power_std = 0.0
        
        # 输出包含功耗的统计信息
        rospy.loginfo("\n性能统计 (跳过前%d帧预热，基于最近 %d 帧):"
                "\n  预处理平均耗时: %.3f秒"
                "\n  推理平均耗时: %.3f秒" 
                "\n  后处理平均耗时: %.3f秒"
                "\n  端到端平均耗时: %.3f秒"
                "\n  实际处理频率: %.2f Hz ± %.2f Hz"
                "\n  平均功耗: %.2f W ± %.2f W",
                self.warmup_frames, len(self.total_times), avg_preprocess, avg_inference,
                avg_postprocess, avg_total, avg_freq, freq_std,
                avg_power, power_std)

    def publish_static_transforms(self):
        """发布静态坐标系变换：FP_ENU0→map 和 FP_POI→base_link"""
        
        # 创建FP_ENU0到map的变换
        enu_to_map = geometry_msgs.msg.TransformStamped()
        enu_to_map.header.stamp = rospy.Time.now()
        enu_to_map.header.frame_id = "FP_ENU0"
        enu_to_map.child_frame_id = "map"
        # 恒等变换
        enu_to_map.transform.rotation.w = 1.0
        
        # 创建FP_POI到base_link的变换
        poi_to_base = geometry_msgs.msg.TransformStamped()
        poi_to_base.header.stamp = rospy.Time.now()
        poi_to_base.header.frame_id = "FP_POI"
        poi_to_base.child_frame_id = "base_link"
        # 恒等变换
        poi_to_base.transform.rotation.w = 1.0
        
        # 一次性发布两个静态变换
        self.static_tf_broadcaster.sendTransform([enu_to_map, poi_to_base])
        
        rospy.loginfo("已发布静态坐标系变换: FP_ENU0→map 和 FP_POI→base_link")
    
    def rgb_callback(self, msg):
        """RGB图像回调，只保存最新的消息"""
        with self.data_lock:
            self.latest_rgb_msg = msg

    def depth_callback(self, msg):
        """深度图像回调，只保存最新的消息"""
        with self.data_lock:
            self.latest_depth_msg = msg

    def pose_callback(self, msg):
        """位姿回调，只保存最新的消息"""
        with self.data_lock:
            self.latest_pose_msg = msg

            # # 从Fixposition消息中提取姿态变换
            # transform = geometry_msgs.msg.TransformStamped()
            # transform.header = msg.header  # 使用相同的时间戳和frame_id
            # transform.child_frame_id = msg.child_frame_id  # 应该是"FP_POI"
            
            # # 复制位置和方向
            # transform.transform.translation.x = msg.pose.pose.position.x
            # transform.transform.translation.y = msg.pose.pose.position.y
            # transform.transform.translation.z = msg.pose.pose.position.z
            # transform.transform.rotation = msg.pose.pose.orientation
            
            # # 发布动态变换
            # tf_broadcaster = tf2_ros.TransformBroadcaster()
            # tf_broadcaster.sendTransform(transform)
    
    def timer_callback(self, event):
        """2Hz定时器回调，处理最新的传感器数据"""
        start_time = time.time()

        with self.data_lock:
            # 检查是否收到了所有必要的消息
            if (self.latest_rgb_msg is None or 
                self.latest_depth_msg is None or 
                self.latest_pose_msg is None):
                rospy.logwarn_throttle(2.0, "等待所有传感器数据就绪...")
                return
            
            # 复制最新消息以便处理
            rgb_msg = self.latest_rgb_msg
            depth_msg = self.latest_depth_msg
            pose_msg = self.latest_pose_msg
        
        try:
            # 转换ROS消息到OpenCV格式
            color_img = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            
            # 获取时间戳
            timestamp = rgb_msg.header.stamp.to_sec()
            # 小车上zed2i的相机内参
            intrinsics = np.array([
                [267.158, 0.0, 311.137],
                [0.0, 267.158, 176.519],
                [0.0, 0.0, 1.0]
            ])

            # 小车上zed2i的相机外参，已根据WSTNet模型数据集修正
            cam2base = np.array([
                [0.0, 0.0, 1.0, 0.08925018],
                [-1.0, 0.0, 0.0, 0.05494124],
                [0.0, -1.0, 0.0, 0.2205098],
                [0.0, 0.0, 0.0, 1.0]
            ])

            # 原始外参
            # cam2base = np.array([
            #     [-0.004017343, -0.9999754686, -0.005737875, 0.0540331795],
            #     [-0.007702066, 0.0057686925, -0.999953699, -0.2201291155],
            #     [0.9999622689, -0.0039729637, -0.007725052, -0.0907319794],
            #     [0.0, 0.0, 0.0, 1.0]
            # ])
            
            utm_easting = pose_msg.pose.pose.position.x
            utm_northing = pose_msg.pose.pose.position.y
            utm_z = pose_msg.pose.pose.position.z

            # 如果是首次处理，设置原点
            if not hasattr(self, 'origin_set') or not self.origin_set:
                self.origin_easting = utm_easting
                self.origin_northing = utm_northing
                self.origin_z = utm_z
                self.origin_set = True

            position = np.array([
                utm_easting - self.origin_easting, 
                utm_northing - self.origin_northing,
                utm_z - self.origin_z
            ])
                    
            quaternion = np.array([
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w
            ])
            
            # 更新当前帧
            self.update_current_frame(color_img, depth_img, timestamp, intrinsics, cam2base, position, quaternion)
            
            # 检查是否有足够的历史帧来进行推理
            if self.frame_buffer.is_full():
                preprocess_start = time.time()  # 预处理开始时间
                
                # 准备推理数据
                inference_data = self.prepare_inference_data()
                
                preprocess_time = time.time() - preprocess_start  # 计算预处理耗时
                if self.frame_counter >= self.warmup_frames:
                    self.preprocessing_times.append(preprocess_time)

                # 添加处理开始时间到数据中
                inference_data['process_start_time'] = start_time
                inference_data['preprocess_time'] = preprocess_time
                inference_data['sensor_stamp'] = timestamp  # RGB图像采集时间(秒)

                # 队列=1 丢旧保新：保证推理总是针对最新窗口，避免积压导致秒级延迟。
                replaced = not self._put_latest(self.inference_queue, inference_data)
                if replaced:
                    rospy.logwarn_throttle(2.0, "推理队列拥塞：已覆盖旧任务（丢旧保新）")

                rospy.loginfo_throttle(2.0, f"以2.5Hz频率传入帧，预处理耗时: {preprocess_time:.3f}秒")

        except CvBridgeError as e:
            rospy.logerr(f"CV桥接错误: {e}")
        except Exception as e:
            rospy.logerr(f"定时器回调处理错误: {str(e)}")
    
    def update_current_frame(self, color_img, depth_img, timestamp, intrinsics, cam2base, position, quaternion):
        """更新当前帧并维护历史帧队列"""
        # 如果已有当前帧，将其添加到历史缓冲区
        if self.current_frame is not None:
            self.frame_buffer.add_frame(
                self.current_frame['color_img'],
                self.current_frame['depth_img'],
                self.current_frame['timestamp'],
                self.current_frame['intrinsics'],
                self.current_frame['cam2base'],
                self.current_frame['position'],
                self.current_frame['quaternion']
            )
        
        # 更新当前帧
        self.current_frame = {
            'color_img': color_img,
            'depth_img': depth_img,
            'timestamp': timestamp,
            'intrinsics': intrinsics,
            'cam2base': cam2base,
            'position': position,
            'quaternion': quaternion
        }
    
    def prepare_inference_data(self):
        """准备推理所需的数据，类似于get_current_data方法"""
        # 获取缓冲区数据
        buffer_data = self.frame_buffer.get_frame_data()
        # 组合历史帧和当前帧
        image_timestamp_list = buffer_data['timestamps'] + [self.current_frame['timestamp']]
        color_img_list = buffer_data['color_imgs'] + [self.current_frame['color_img']]
        depth_img_list = buffer_data['depth_imgs'] + [self.current_frame['depth_img']]
        intrinsics_list = buffer_data['intrinsics'] + [self.current_frame['intrinsics']]
        cam2base_list = buffer_data['cam2base'] + [self.current_frame['cam2base']]
        
        # 构建滑动窗口的状态数据字典
        position_list = buffer_data['positions'] + [self.current_frame['position']]
        quaternion_list = buffer_data['quaternions'] + [self.current_frame['quaternion']]
        timestamp_list = buffer_data['timestamps'] + [self.current_frame['timestamp']]

        rosbag_dict = {
            'states_timestamp': timestamp_list,
            'position': position_list,
            'quaternion': quaternion_list
        }
        
        # 处理数据
        extrinsics_list = self.get_extrinsics(rosbag_dict, image_timestamp_list, cam2base_list)
        pcloud_data = self.read_pcloud(depth_img_list, intrinsics_list, extrinsics_list)
        
        # 使用PyTorch在GPU上处理图像和深度
        device = self.device
        depth_target_list = []
        color_tensor_list = []

        # 预先计算下采样后的深度尺寸
        fW = int(np.ceil(self.config.MODEL.INPUT_SIZE[0] / self.config.MODEL.DOWNSAMPLE))
        fH = int(np.ceil(self.config.MODEL.INPUT_SIZE[1] / self.config.MODEL.DOWNSAMPLE))

        db0, db1, db2 = self.grid_bounds['dbound']
        n_d = int((db1 - db0) / db2)

        for t in range(self.n_frames):
            color_img = color_img_list[t]
            depth_img = depth_img_list[t]

            # 记录原始尺寸用于内参缩放
            orig_h, orig_w = color_img.shape[0], color_img.shape[1]

            # 转为GPU tensor
            color = torch.from_numpy(color_img).to(device=device, dtype=torch.float32)  # H,W,C
            color = color.permute(2, 0, 1) / 255.0  # C,H,W

            depth = torch.from_numpy(depth_img).to(device=device, dtype=torch.float32)  # H,W

            # 尺寸调整到网络输入尺寸
            target_w, target_h = self.config.MODEL.INPUT_SIZE  # (W, H)
            if (target_w != orig_w) or (target_h != orig_h):
                # 缩放内参（仍在CPU上，开销极小）
                intrinsics_list[t][0, :] *= (target_w / float(orig_w))
                intrinsics_list[t][1, :] *= (target_h / float(orig_h))

                # 颜色图双线性插值
                color = F.interpolate(color.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)[0]
                # 深度图最近邻插值
                depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='nearest')[0, 0]

            color_tensor_list.append(color)

            # 深度图处理：去除无效值
            depth = depth.clone()
            depth[~torch.isfinite(depth)] = 0.0

            # 下采样到体素分辨率
            depth_down = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(fH, fW), mode='nearest')[0, 0]

            # 量化深度值为索引
            depth_idx = torch.round((depth_down - db0) / db2).to(torch.int64)

            if not self.config.MODEL.PREDICT_DEPTH:
                # 创建深度体素网格 (n_d, fH, fW)
                xs = torch.arange(0, fH, device=device)
                ys = torch.arange(0, fW, device=device)
                xs, ys = torch.meshgrid(xs, ys, indexing='ij')

                d_flat = depth_idx.view(-1)
                xs_flat = xs.view(-1)
                ys_flat = ys.view(-1)

                valid = (d_flat >= 0) & (d_flat <= n_d - 1)
                d_flat = d_flat[valid]
                xs_flat = xs_flat[valid]
                ys_flat = ys_flat[valid]

                depth_voxel = torch.zeros((n_d, fH, fW), device=device, dtype=torch.float32)
                if d_flat.numel() > 0:
                    depth_voxel[d_flat, xs_flat, ys_flat] = 1.0
                depth_target = depth_voxel
            else:
                # 预测深度模式，仅限制深度范围
                depth_target = torch.clamp(depth_idx, 0, n_d - 1)

            depth_target_list.append(depth_target)

        # 转换为tensor/ndarray并添加批次维度
        color_img = torch.stack(color_tensor_list)  # (T, C, H, W) on device
        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        depth_target = torch.stack(depth_target_list)  # (T, ...) on device

        # 添加批次维度
        color_img = color_img.unsqueeze(0)  # (1, T, C, H, W)
        pcloud_data = pcloud_data.unsqueeze(0).type(torch.float32)  # already on device
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).type(torch.float32)
        extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).type(torch.float32)
        depth_target = depth_target.unsqueeze(0)
        
        # 记录当前ROS时间戳，用于发布消息
        ros_timestamp = rospy.Time.now()
        
        return {
            'color_img': color_img,
            'pcloud': pcloud_data,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'depth_target': depth_target,
            'ros_timestamp': ros_timestamp
        }
    
    def get_extrinsics(self, rosbag_dict, image_timestamp, cam2base_list):
        """获取相机外参矩阵列表，目前仍在CPU"""
        sync = [bisect.bisect_left(rosbag_dict['states_timestamp'], i) for i in image_timestamp]
        sync = [min(i, len(rosbag_dict['states_timestamp'])-1) for i in sync]
        
        position = np.asarray(rosbag_dict['position'])[sync]
        quaternion = np.asarray(rosbag_dict['quaternion'])[sync]
        
        # 四元数转换为旋转矩阵
        quaternion = quaternion[:, [0,1,2,3]]  # 确保适应scipy Rotation的格式
        rotation = R.from_quat(quaternion)
        
        # 最后一帧作为参考坐标系
        last_rotation = rotation[-1]
        last_position = position[-1]
        
        extrinsics_list = []
        for i, cam2base in enumerate(cam2base_list):
            # 计算当前帧相对于最后一帧的变换矩阵
            current_rotation = rotation[i]
            current_position = position[i]
            
            # 相对旋转
            relative_rotation = last_rotation.inv() * current_rotation
            relative_rotation_matrix = relative_rotation.as_matrix()
            
            # 相对平移
            relative_position = current_position - last_position
            relative_position = last_rotation.inv().apply(relative_position)
            
            # 构建相对变换矩阵
            relative_transform = np.eye(4)
            relative_transform[:3, :3] = relative_rotation_matrix
            relative_transform[:3, 3] = relative_position
            
            # 计算最终的外参矩阵
            extrinsics = relative_transform @ cam2base
            extrinsics_list.append(extrinsics)
                
        return extrinsics_list
    
    def read_pcloud(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        """使用PyTorch GPU加速的点云处理函数"""
        device = self.device
        depth_images = torch.tensor(np.stack(depth_image_list), dtype=torch.float32, device=device)
        cam_intrinsics = torch.tensor(np.stack(cam_intrinsics_list), dtype=torch.float32, device=device)
        cam_extrinsics = torch.tensor(np.stack(cam_extrinsics_list), dtype=torch.float32, device=device)
        batch_size, height, width = depth_images.shape
        dx = torch.tensor([row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]], 
                        device=device)
        cx = torch.tensor([round(row[1]/row[2] - 0.5) for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]],
                        device=device).long()
        nx = torch.tensor([(row[1] - row[0]) / row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]],
                        device=device).long()
        y_grid = torch.arange(0, height, device=device)
        x_grid = torch.arange(0, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size, -1)
        y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size, -1)
        depth_flat = depth_images.reshape(batch_size, -1)
        temporal_grid = torch.zeros((batch_size, nx[2], nx[0], nx[1]), device=device)
        for t in range(batch_size):
            points = torch.stack([
                x_grid[t] * depth_flat[t],
                y_grid[t] * depth_flat[t],
                depth_flat[t]
            ], dim=1)

            valid_mask = torch.isfinite(points[:, 2]) & (points[:, 2] > 0)
            points = points[valid_mask]
            
            if points.shape[0] == 0:
                continue

            rotation = cam_extrinsics[t, :3, :3]
            translation = cam_extrinsics[t, :3, 3]
            cam_inv = torch.inverse(cam_intrinsics[t])
            combined_transform = rotation @ cam_inv
            points = (combined_transform @ points.t()).t() + translation
            
            idx_lidar = torch.round(cx.unsqueeze(0) - points / dx.unsqueeze(0) - 0.5).long()
            
            valid_idx = ((idx_lidar[:, 0] >= 0) & (idx_lidar[:, 0] < nx[0]) &
                        (idx_lidar[:, 1] >= 0) & (idx_lidar[:, 1] < nx[1]) &
                        (idx_lidar[:, 2] >= 0) & (idx_lidar[:, 2] < nx[2]))
            
            idx_lidar = idx_lidar[valid_idx]
            
            if idx_lidar.shape[0] == 0:
                continue
            
            # 为了避免重复索引的问题，使用更高效的方式
            # 使用sparse_coo_tensor更高效，但实现更复杂
            # 这里使用简单的方法，对每个点填充体素
            grid = torch.zeros((nx[0], nx[1], nx[2]), device=device)
            grid[idx_lidar[:, 0], idx_lidar[:, 1], idx_lidar[:, 2]] = 1
            temporal_grid[t] = grid.permute(2, 0, 1)
        
        # 直接返回GPU上的体素网格张量 (T, Z, X, Y)
        return temporal_grid
    
    def inference_loop(self):
        """推理线程主循环"""
        with torch.no_grad():
            while not rospy.is_shutdown():
                try:
                    # 从队列获取数据
                    if self.inference_queue.empty():
                        rospy.sleep(0.01)  # 短暂休眠以减少CPU使用率
                        continue
                    
                    # 获取推理数据
                    data = self.inference_queue.get()
                    process_start_time = data['process_start_time']
                    preprocess_time = data['preprocess_time']

                    inference_start = time.time()  # 记录开始时间
                    color_img = data['color_img'].to(self.device)
                    pcloud = data['pcloud'].to(self.device)
                    intrinsics = data['intrinsics'].to(self.device)
                    extrinsics = data['extrinsics'].to(self.device)
                    depth_target = data['depth_target'].to(self.device)
                    ros_timestamp = data['ros_timestamp']
                    sensor_stamp = data.get('sensor_stamp', None)

                    # 根据配置选择是否启用CUDA AMP混合精度
                    amp_ctx = torch.cuda.amp.autocast if (self.device.type == 'cuda' and self.use_amp) else nullcontext

                    # 执行推理，trav_map 形状为 (1, 2, H, W)，safety_map 与单通道trav_map格式相同
                    if self.safety_head:
                        with amp_ctx():
                            trav_map, safety_map, depth_logits, debug = self.model(
                                color_img, pcloud, intrinsics, extrinsics, depth_target
                            )
                    else:
                        with amp_ctx():
                            trav_map, depth_logits, debug = self.model(
                                color_img, pcloud, intrinsics, extrinsics, depth_target
                            )
                        _, _, H, W = trav_map.shape
                        safety_map = torch.ones((1, 1, H, W), device=self.device, dtype=torch.float32)

                    inference_time = time.time() - inference_start
                    if self.frame_counter >= self.warmup_frames:
                        self.inference_times.append(inference_time)

                    # 构造结果并放入结果队列
                    result_data = {
                        'trav_map': trav_map.cpu().numpy(),
                        'safety_map': safety_map.cpu().numpy(),
                        'original_img': color_img[0, -1].permute(1, 2, 0).cpu().numpy() * 255,
                        'ros_timestamp': ros_timestamp,
                        'sensor_stamp': sensor_stamp,
                        'process_start_time': process_start_time,
                        'preprocess_time': preprocess_time,
                        'inference_time': inference_time,
                        'postprocess_start_time': time.time()
                    }

                    # 将结果放入结果队列（只保留最新结果，避免发布线程落后时继续积压旧结果）
                    replaced = not self._put_latest(self.result_queue, result_data)
                    if replaced:
                        rospy.logwarn_throttle(2.0, "结果队列拥塞：已覆盖旧结果（丢旧保新）")
                    
                except Exception as e:
                    rospy.logerr(f"推理线程错误: {e}")

    def publishing_loop(self):
        """发布线程主循环"""
        while not rospy.is_shutdown():
            try:
                if self.result_queue.empty():
                    rospy.sleep(0.01)
                    continue
                
                result = self.result_queue.get()
                trav_map_np = result['trav_map']
                safety_map_np = result['safety_map']
                original_img = result['original_img']
                ros_timestamp = result['ros_timestamp']
                process_start_time = result['process_start_time']
                preprocess_time = result['preprocess_time']
                inference_time = result['inference_time']
                postprocess_start_time = result['postprocess_start_time']
                sensor_stamp = result.get('sensor_stamp', None)

                if sensor_stamp is not None:
                    age = time.time() - float(sensor_stamp)
                    rospy.loginfo_throttle(2.0, f"端到端延迟(sensor->pub): {age:.3f}s, inference={inference_time:.3f}s")

                # 对trav和safety均进行转置和旋转处理，使其朝向与RViz中的预期方向一致
                corrected_trav_map_np = trav_map_np.copy()
                for i in range(trav_map_np.shape[1]):
                    corrected_trav_map_np[0, i] = np.flipud(np.rot90(trav_map_np[0, i], k=-1))
                corrected_safety_map_np = safety_map_np.copy()
                for i in range(safety_map_np.shape[1]):
                    corrected_safety_map_np[0, i] = np.flipud(np.rot90(safety_map_np[0, i], k=-1))
                
                # 使用校正后的数据
                trav_map_msg = self.create_traversability_map_msg(corrected_trav_map_np, ros_timestamp)
                self.trav_map_pub.publish(trav_map_msg)
                safety_map_msg = self.create_safety_map_msg(corrected_safety_map_np, ros_timestamp)
                self.safety_map_pub.publish(safety_map_msg)
                
                mu_grid_msg = self.create_occupancy_grid_msg(corrected_trav_map_np[0, 0], ros_timestamp, "mu")
                nu_grid_msg = self.create_occupancy_grid_msg(corrected_trav_map_np[0, 1], ros_timestamp, "nu")
                safety_grid_msg = self.create_occupancy_grid_msg(corrected_safety_map_np[0, 0], ros_timestamp, "safety")
                
                self.trav_viz_mu_pub.publish(mu_grid_msg)
                self.trav_viz_nu_pub.publish(nu_grid_msg)
                self.safety_viz_pub.publish(safety_grid_msg)

                # 发布mu、nu以及safety的图像可视化
                self.publish_image_messages(corrected_trav_map_np, corrected_safety_map_np, ros_timestamp)
                
                # 记录发布频率
                rospy.logdebug("通过性地图已发布")

                # 计算后处理耗时
                postprocess_time = time.time() - postprocess_start_time
                self.frame_counter += 1
                if self.frame_counter >= self.warmup_frames:
                    self.postprocessing_times.append(postprocess_time)
                    
                    # 计算总耗时
                    total_time = time.time() - process_start_time
                    self.total_times.append(total_time)
                    
                    # 计算处理频率
                    current_time = time.time()
                    if self.last_frame_time is not None:
                        freq = 1.0 / (current_time - self.last_frame_time)
                        self.processing_freqs.append(freq)
                    self.last_frame_time = current_time
                else:
                    # 预热阶段的日志
                    rospy.loginfo_throttle(1.0, f"预热阶段: 帧 {self.frame_counter}/{self.warmup_frames}, 跳过统计收集")
                    # 更新时间戳但不计入统计
                    self.last_frame_time = time.time()

                # 记录详细的时间信息
                # rospy.loginfo_throttle(2.0, f"帧处理完成: 预处理={preprocess_time:.3f}秒, 推理={inference_time:.3f}秒, "
                #                     f"后处理={postprocess_time:.3f}秒, 总计={total_time:.3f}秒")
                
            except Exception as e:
                rospy.logerr(f"发布线程错误: {e}")
    
    def create_traversability_map_msg(self, trav_map_np, timestamp):
        """创建自定义通过性地图消息"""
        msg = TraversabilityMap()
        msg.header.stamp = timestamp
        msg.header.frame_id = "base_link"
        
        # 获取通过性地图的维度
        trav_map_mu = trav_map_np[0, 0]  # 第一个通道是mu
        trav_map_nu = trav_map_np[0, 1]  # 第二个通道是nu
        
        # 设置地图尺寸
        msg.width = trav_map_mu.shape[1]
        msg.height = trav_map_mu.shape[0]
        
        # 设置分辨率（米/像素）
        msg.resolution = self.grid_bounds['xbound'][2]  # 使用x方向的分辨率
        
        # 设置原点（相对于base_link的位置）
        # 地图的原点在左下角，需要将其转换为ROS坐标系中的位置
        x_origin = self.grid_bounds['xbound'][0]  # 通常为-2.0（后方2米）
        y_origin = self.grid_bounds['ybound'][0]  # 通常为-5.0（左侧5米）
        
        msg.origin.position.x = x_origin
        msg.origin.position.y = y_origin
        msg.origin.position.z = 0.0
        
        # 设置原点的方向（默认不旋转）
        msg.origin.orientation.w = 1.0
        
        # 填充数据
        msg.mu_data = trav_map_mu.flatten().tolist()
        msg.nu_data = trav_map_nu.flatten().tolist()
        
        return msg

    def create_safety_map_msg(self, safety_map_np, timestamp):
        """创建自定义安全系数地图消息"""
        msg = SafetyMap()
        msg.header.stamp = timestamp
        msg.header.frame_id = "base_link"

        # safety_map 目前假设为单通道，与mu/nu单通道格式相同
        safety_channel = safety_map_np[0, 0]

        # 设置地图尺寸
        msg.width = safety_channel.shape[1]
        msg.height = safety_channel.shape[0]

        # 设置分辨率（米/像素）
        msg.resolution = self.grid_bounds['xbound'][2]

        # 设置原点（相对于base_link的位置）
        x_origin = self.grid_bounds['xbound'][0]
        y_origin = self.grid_bounds['ybound'][0]

        msg.origin.position.x = x_origin
        msg.origin.position.y = y_origin
        msg.origin.position.z = 0.0

        msg.origin.orientation.w = 1.0

        # 填充数据
        msg.safety_data = safety_channel.flatten().tolist()

        return msg
    
    def create_occupancy_grid_msg(self, trav_map_channel, timestamp, channel_name):
        """创建OccupancyGrid消息用于可视化"""
        msg = OccupancyGrid()
        msg.header.stamp = timestamp
        msg.header.frame_id = "base_link"
        
        # 设置地图尺寸
        msg.info.width = trav_map_channel.shape[1]
        msg.info.height = trav_map_channel.shape[0]
        
        # 设置分辨率（米/像素）
        msg.info.resolution = self.grid_bounds['xbound'][2]
        
        # 设置原点
        x_origin = self.grid_bounds['xbound'][0]
        y_origin = self.grid_bounds['ybound'][0]
        
        msg.info.origin.position.x = x_origin
        msg.info.origin.position.y = y_origin
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # 将float值转换为占用值（0-100）
        # 对于mu和nu通道都是值越大表示可通过性越好，转换为占用值时应该是值越大占用值越小
        # 归一化到0-1
        norm_data = (trav_map_channel - trav_map_channel.min()) / (trav_map_channel.max() - trav_map_channel.min() + 1e-6)
        # 反转并缩放到0-100
        occupancy_data = (1.0 - norm_data) * 100
        
        # 转换为整数并保存为列表
        msg.data = occupancy_data.astype(np.int8).flatten().tolist()
        
        return msg

    def publish_image_messages(self, trav_map_np, safety_map_np, ros_timestamp):
        """在publishing_loop方法中添加这个调用"""
        
        # 创建图像消息发布器(如果尚未创建)
        if not hasattr(self, 'trav_mu_img_pub'):
            self.trav_mu_img_pub = rospy.Publisher('/traversability_map_viz/mu_image', Image, queue_size=1)
            self.trav_nu_img_pub = rospy.Publisher('/traversability_map_viz/nu_image', Image, queue_size=1)
            self.safety_img_pub = rospy.Publisher('/safety_map_viz/safety_image', Image, queue_size=1)
        
        # 获取当前结果
        # result = self.result_queue.get()
        # trav_map_np = result['trav_map']
        # ros_timestamp = result['ros_timestamp']
        
        # 提取mu和nu通道
        mu_map = trav_map_np[0, 0]  # 第一个通道是mu
        nu_map = trav_map_np[0, 1]  # 第二个通道是nu
        safety_map = safety_map_np[0, 0]
        
        # 保持原始方向不变 - 这与matplotlib显示的方向相同
        # 转换为8位无符号整型(0-255)，显示为灰度图
        mu_img = (mu_map * 255).astype(np.uint8)
        nu_img = (nu_map * 255).astype(np.uint8)
        safety_img = (safety_map * 255).astype(np.uint8)
        
        # jet图显示
        # mu_img_color = cv2.applyColorMap(mu_img, cv2.COLORMAP_JET)
        # nu_img_color = cv2.applyColorMap(nu_img, cv2.COLORMAP_JET)
        # safety_img_color = cv2.applyColorMap(safety_img, cv2.COLORMAP_JET)
        
        # 转换为ROS图像消息
        try:
            mu_img_msg = self.bridge.cv2_to_imgmsg(mu_img, encoding="mono8")
            nu_img_msg = self.bridge.cv2_to_imgmsg(nu_img, encoding="mono8")
            safety_img_msg = self.bridge.cv2_to_imgmsg(safety_img, encoding="mono8")
            
            # jet图显示
            # mu_img_msg = self.bridge.cv2_to_imgmsg(mu_img_color, encoding="bgr8")
            # nu_img_msg = self.bridge.cv2_to_imgmsg(nu_img_color, encoding="bgr8")
            
            # 设置时间戳和坐标系
            mu_img_msg.header.stamp = ros_timestamp
            mu_img_msg.header.frame_id = "base_link"
            nu_img_msg.header.stamp = ros_timestamp
            nu_img_msg.header.frame_id = "base_link"
            safety_img_msg.header.stamp = ros_timestamp
            safety_img_msg.header.frame_id = "base_link"
            
            # 发布消息
            self.trav_mu_img_pub.publish(mu_img_msg)
            self.trav_nu_img_pub.publish(nu_img_msg)
            self.safety_img_pub.publish(safety_img_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误: {e}")

    @staticmethod
    def _put_latest(q: queue.Queue, item) -> bool:
        """向队列放入 item；若队列满则丢弃最旧元素，再放入新的。

        返回:
            True: 直接放入成功（未发生覆盖）
            False: 发生覆盖（丢弃了旧元素）
        """
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            try:
                q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                # 极端竞争条件下仍可能满；此时直接放弃本次数据
                return False
            return False
        
if __name__ == '__main__':
    try:
        node = TraversabilityInferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass