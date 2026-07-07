import os
import cv2
import torch
import bisect
import numpy as np
import pandas as pd
import torch.utils.data as DataLoader
from itertools import compress
from scipy.spatial.transform import Rotation as R
from collections import deque

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

class InputFolderDataset:
    """
    从inputs文件夹加载数据的数据集类，按顺序处理图像并维护滑动窗口
    """
    def __init__(self, configs, input_folder='inputs'):
        """
        初始化输入文件夹数据集
        
        Args:
            configs: 配置对象
            input_folder: 输入数据目录
        """
        print(f"Initializing dataset from {input_folder}...")
        self.dt = configs.TRAINING.DT
        self.image_size = configs.MODEL.INPUT_SIZE
        self.downsample = configs.MODEL.DOWNSAMPLE
        self.grid_bounds = configs.MODEL.GRID_BOUNDS
        self.n_frames = configs.MODEL.TIME_LENGTH  # 6
        self.predict_depth = configs.MODEL.PREDICT_DEPTH
        # self.bin_width = 0.2
        self.dtype = torch.float32
        self.input_folder = input_folder
        
        # 创建帧缓冲区
        self.buffer = FrameBuffer(buffer_size=self.n_frames-1)  # 5帧历史
        
        # 当前帧数据
        self.current_frame = {
            'color_img': None,
            'depth_img': None,
            'timestamp': None,
            'intrinsics': None,
            'cam2base': None,
            'position': None,
            'quaternion': None
        }
        
        # 读取输入数据
        self.load_input_data()
        
        # 当前数据位置跟踪
        self.current_img_idx = 0
        
        # 网格尺寸计算
        self.map_size = (
            int((self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0])/self.grid_bounds['ybound'][2]))
        
        print("Dataset initialized!")

    def load_input_data(self):
        """
        加载inputs文件夹中的数据，使用与dataloader.py相同的解析方法
        """
        # 读取images.csv文件
        images_path = os.path.join(self.input_folder, 'images.csv')
        if not os.path.exists(images_path):
            images_path = os.path.join(self.input_folder, 'image.csv')
        
        self.images_data = pd.read_csv(images_path)
        
        # 读取states.csv文件
        states_path = os.path.join(self.input_folder, 'states.csv')
        self.states_data = pd.read_csv(states_path)
        
        # 提取数据
        image_timestamp_list = []
        color_fname_list = []
        depth_fname_list = []
        intrinsics_list = []
        cam2base_list = []
        
        # 与dataloader.py保持一致的解析方法
        map_float = lambda x: np.array(list(map(float, x)))
        
        # 解析图像数据
        for i in range(len(self.images_data)):
            try:
                timestamp = self.images_data['timestamp'].iloc[i]
                color_fname = self.images_data['image'].iloc[i]
                depth_fname = self.images_data['depth'].iloc[i]
                intrinsics = self.images_data['intrinsics'].iloc[i]
                cam2base = self.images_data['cam2base'].iloc[i]
                
                # 构建完整路径
                color_path = os.path.join(self.input_folder, color_fname)
                depth_path = os.path.join(self.input_folder, depth_fname)
                
                # 使用与dataloader.py一致的解析方法但更健壮
                intrinsics = intrinsics.replace('[', ' ').replace(']', ' ')
                # 移除所有逗号和其他可能干扰的字符
                intrinsics = intrinsics.replace(',', ' ')
                intrinsics = intrinsics.split()
                intrinsics = map_float(intrinsics).reshape((3,3))
                
                cam2base = cam2base.replace('[', ' ').replace(']', ' ')
                # 同样移除逗号
                cam2base = cam2base.replace(',', ' ')
                cam2base = cam2base.split()
                cam2base = map_float(cam2base).reshape((4,4))
                
                image_timestamp_list.append(timestamp)
                color_fname_list.append(color_path)
                depth_fname_list.append(depth_path)
                intrinsics_list.append(intrinsics)
                cam2base_list.append(cam2base)
                
            except Exception as e:
                print(f"警告：处理图像数据第{i}行时出错，跳过此帧: {e}")
        
        # 解析状态数据
        states_timestamp_list = []
        position_list = []
        quaternion_list = []
        
        for i in range(len(self.states_data)):
            try:
                timestamp, position, quaternion, *other_cols = self.states_data.iloc[i]
                
                # 与dataloader.py保持一致的解析方法
                position = position[1:-1].split()
                position = map_float(position)
                # NOTE:测试集时使用
                position[0] *= 1e3
                position[1] *= 1e3
                
                quaternion = quaternion[1:-1].split()
                quaternion = map_float(quaternion)
                
                states_timestamp_list.append(timestamp)
                position_list.append(position)
                quaternion_list.append(quaternion)
                
            except Exception as e:
                print(f"警告：处理状态数据第{i}行时出错，跳过此行: {e}")
        
        # 使用dataloader.py中的过滤方式
        idxs = np.asarray(image_timestamp_list) < states_timestamp_list[-1]
        self.image_timestamps = list(compress(image_timestamp_list, idxs))
        self.img_paths = list(compress(color_fname_list, idxs))
        self.depth_paths = list(compress(depth_fname_list, idxs))
        self.intrinsics_list = list(compress(intrinsics_list, idxs))
        self.cam2base_list = list(compress(cam2base_list, idxs))
        
        self.state_timestamps = states_timestamp_list
        self.positions = position_list
        self.quaternions = quaternion_list
        
        print(f"加载了 {len(self.image_timestamps)} 张有效图像和 {len(self.state_timestamps)} 个状态")


    def get_next_frame(self):
        """
        获取下一帧图像和对应的状态数据 - 简化状态查找
        
        Returns:
            bool: 是否成功获取新帧
        """
        if self.current_img_idx >= len(self.image_timestamps):
            print("End of dataset reached")
            return False
        
        # 获取当前图像数据
        timestamp = self.image_timestamps[self.current_img_idx]
        img_path = self.img_paths[self.current_img_idx]
        depth_path = self.depth_paths[self.current_img_idx]
        intrinsics = self.intrinsics_list[self.current_img_idx]
        cam2base = self.cam2base_list[self.current_img_idx]
        
        # 加载图像
        color_img = cv2.imread(img_path, -1)
        if color_img is None:
            print(f"Warning: Failed to load image at {img_path}")
            self.current_img_idx += 1
            return self.get_next_frame()
            
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = np.clip(color_img * 1.2, 0, 255).astype(np.uint8)
        depth_img = cv2.imread(depth_path, -1)
        
        if depth_img is None:
            print(f"Warning: Failed to load depth image at {depth_path}")
            self.current_img_idx += 1
            return self.get_next_frame()
        
        # 查找对应的状态数据 - 使用更简洁的方式
        state_idx = bisect.bisect_left(self.state_timestamps, timestamp)
        state_idx = min(state_idx, len(self.state_timestamps)-1)
        
        # 如果不是精确匹配，找到最近的时间戳
        if state_idx > 0 and (timestamp != self.state_timestamps[state_idx]):
            if (timestamp - self.state_timestamps[state_idx-1]) < (self.state_timestamps[state_idx] - timestamp):
                state_idx = state_idx - 1
                
        position = self.positions[state_idx]
        quaternion = self.quaternions[state_idx]
        
        # 更新当前帧
        self.update_current_frame(
            color_img, depth_img, timestamp, intrinsics, cam2base, position, quaternion
        )
        
        # 移动到下一帧
        self.current_img_idx += 1
        return True
    
    def update_current_frame(self, color_img, depth_img, timestamp, intrinsics, cam2base, position, quaternion):
        """
        更新当前帧并维护历史帧队列
        """
        # 如果已有当前帧，将其添加到历史缓冲区
        if self.current_frame['color_img'] is not None:
            self.buffer.add_frame(
                self.current_frame['color_img'],
                self.current_frame['depth_img'],
                self.current_frame['timestamp'],
                self.current_frame['intrinsics'],
                self.current_frame['cam2base'],
                self.current_frame['position'],
                self.current_frame['quaternion']
            )
        
        # 更新当前帧
        self.current_frame['color_img'] = color_img
        self.current_frame['depth_img'] = depth_img
        self.current_frame['timestamp'] = timestamp
        self.current_frame['intrinsics'] = intrinsics
        self.current_frame['cam2base'] = cam2base
        self.current_frame['position'] = position
        self.current_frame['quaternion'] = quaternion
        
        return self.buffer.is_full()
    
    def get_current_data(self):
        """
        获取当前滑动窗口中推理所需的所有数据，包括color_img, pcloud_data, intrinsics, extrinsics, depth_target
        """
        # 获取缓冲区数据
        buffer_data = self.buffer.get_frame_data()
        
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
        
        # 处理深度图
        depth_target_list = []
        for t in range(self.n_frames):
            color_img = color_img_list[t]
            depth_img = depth_img_list[t]
            
            # 尺寸调整
            if (self.image_size[0] != color_img.shape[1]) or (self.image_size[1] != color_img.shape[0]):
                intrinsics_list[t][0, :] *= (self.image_size[0]/color_img.shape[1])
                intrinsics_list[t][1, :] *= (self.image_size[1]/color_img.shape[0])
                color_img = cv2.resize(color_img, self.image_size, interpolation=cv2.INTER_AREA)
                depth_img = cv2.resize(depth_img, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # 转换为tensor格式
            color_img_list[t] = torch.from_numpy(color_img).permute(2,0,1).type(self.dtype) / 255.0
            
            # 深度图处理
            # NOTE:scout的深度图使用
            # depth_target = depth_img * 1e-3  # 转换为米
            depth_target = depth_img.astype(np.float32)  # scout的zed相机深度值单位本就为m，不用乘以1e-3
            depth_target[~np.isfinite(depth_target)] = 0
            depth_size = (int(np.ceil(self.image_size[0]/self.downsample)), 
                        int(np.ceil(self.image_size[1]/self.downsample)))
            depth_target = cv2.resize(depth_target, depth_size, interpolation=cv2.INTER_NEAREST)
            
            # 量化深度值
            depth_target = np.round((depth_target - self.grid_bounds['dbound'][0]) / 
                                    self.grid_bounds['dbound'][2]).astype('int')
            
            n_d = int((self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0]) / 
                    self.grid_bounds['dbound'][2])
            
            # 深度图格式处理
            if not self.predict_depth:
                # 计算下采样后的深度图像尺寸
                fH = int(np.ceil(self.image_size[1] / self.downsample))
                fW = int(np.ceil(self.image_size[0] / self.downsample))
                
                # 创建网格坐标
                xs = np.arange(0, fH)
                ys = np.arange(0, fW)
                xs, ys = np.meshgrid(xs, ys, indexing='ij')
                
                # 将深度图像和网格坐标堆叠
                points = np.stack((
                    depth_target.flatten(),  # 深度索引
                    xs.flatten(),            # 高度坐标
                    ys.flatten()), -1)       # 宽度坐标
                
                # 创建深度体素网格
                depth_voxel = np.zeros((n_d, fH, fW))
                idxs = (points[:,0] >= 0) & (points[:,0] <= n_d-1)
                points = points[idxs]
                if len(points) > 0:  # 确保有有效点
                    depth_voxel[points[:,0], points[:,1], points[:,2]] = 1.0
                depth_target = depth_voxel.astype('float32')
            else:
                # 预测深度模式，仅限制深度范围
                depth_target = np.clip(depth_target, 0, n_d-1)
            
            depth_target_list.append(depth_target)
            
        # 转换为tensor
        color_img = torch.stack(color_img_list)
        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        depth_target = np.stack(depth_target_list)
        
        # 添加批次维度
        color_img = color_img.unsqueeze(0)
        pcloud_data = torch.from_numpy(pcloud_data).unsqueeze(0).type(self.dtype)
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).type(self.dtype)
        extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).type(self.dtype)
        depth_target = torch.from_numpy(depth_target).unsqueeze(0)
        
        return color_img, pcloud_data, intrinsics, extrinsics, depth_target
    
    def get_extrinsics(self, rosbag_dict, image_timestamp, cam2base_list):
        """
        获取相机外参 - 使用dataloader.py的实现
        """
        # sync = [bisect.bisect_left(rosbag_dict['states_timestamp'], i) for i in image_timestamp]
        # # 确保索引在有效范围内
        # sync = [min(i, len(rosbag_dict['states_timestamp'])-1) for i in sync]
        
        # position = np.asarray(rosbag_dict['position'])[sync]
        # quaternion = np.asarray(rosbag_dict['quaternion'])[sync]
        
        # quaternion = quaternion[:, [1,2,3,0]]
        # rotation = R.from_quat(quaternion)
        # euler_angle = rotation.as_euler('zyx')
        
        # # 使用最后一帧作为参考
        # heading_rot = R.from_euler('zyx', [euler_angle[-1,0], 0, 0])
        # position = (heading_rot.inv().as_matrix() @ (position.T - position[-1,None].T)).T

        # extrinsics_list = []
        # for i, cam2base in enumerate(cam2base_list):
        #     base_rot = R.from_euler('zyx', [0, euler_angle[i,1], euler_angle[i,2]])
        #     base_trans = np.eye(4)
        #     base_trans[:3,:3] = base_rot.as_matrix()
            
        #     odom_rot = R.from_euler('zyx', [euler_angle[i,0]-euler_angle[-1,0], 0, 0])
        #     odom_trans = np.eye(4)
        #     odom_trans[:3,:3] = odom_rot.as_matrix()
        #     odom_trans[:2,3] = position[i,:2]
            
        #     extrinsics = odom_trans @ base_trans @ cam2base
        #     extrinsics_list.append(extrinsics)

        # 直接使用四元数，不使用欧拉角，避免万向锁问题
        # NOTE:测试集时使用
        sync = [bisect.bisect_left(rosbag_dict['states_timestamp'], i) for i in image_timestamp]
        sync = [min(i, len(rosbag_dict['states_timestamp'])-1) for i in sync]
        
        position = np.asarray(rosbag_dict['position'])[sync]
        quaternion = np.asarray(rosbag_dict['quaternion'])[sync]
        
        # 四元数转换为旋转矩阵
        quaternion = quaternion[:, [1,2,3,0]]  # 适应scipy Rotation的格式
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
            # 世界坐标系 -> 相对于最后一帧的坐标系 -> 当前帧的相机坐标系
            extrinsics = relative_transform @ cam2base
            extrinsics_list.append(extrinsics)
                
        return extrinsics_list
    
    def read_pcloud(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        """
        将深度图转换为体素网格，即深度点云，并用相机内参和外参从相机坐标系转换到世界坐标系

        Args:
            depth_image_list (list): 深度图像列表
            cam_intrinsics_list (list): 相机内参矩阵列表
            cam_extrinsics_list (list): 相机外参矩阵列表

        Returns:
            numpy.ndarray: 沿时间轴堆叠的深度体素网格
        """
        temporal_grid = []         
        for t in range(len(depth_image_list)):
            depth_image = depth_image_list[t]
            cam_intrinsics = cam_intrinsics_list[t]
            cam_extrinsics = cam_extrinsics_list[t]
            rotation, translation = cam_extrinsics[:3, :3], cam_extrinsics[:3, 3]
            xs = np.arange(0, depth_image.shape[1]) 
            ys = np.arange(0, depth_image.shape[0]) 
            xs, ys = np.meshgrid(xs, ys)
            # wayfaster数据集深度值单位为mm
            # points = np.stack((
            #     xs.flatten() * depth_image.flatten() * 1e-3,
            #     ys.flatten() * depth_image.flatten() * 1e-3,
            #     depth_image.flatten() * 1e-3), -1)
            # scout的zed深度值单位为m
            # NOTE:cout的zed采集的数据集时使用
            points = np.stack((
                xs.flatten() * depth_image.flatten(),
                ys.flatten() * depth_image.flatten(),
                depth_image.flatten()), -1)
            # print(f"原始点数: {len(points)}")
            points = points[np.isfinite(points[:,2])]
            # print(f"过滤非有限值后点数: {len(points)}")
            points = points[points[:,2] > 0]
            # print(f"过滤零深度后点数: {len(points)}")
            # print(f"转换前点云范围: x({np.min(points[:,0]):.2f}-{np.max(points[:,0]):.2f}), "
                # f"y({np.min(points[:,1]):.2f}-{np.max(points[:,1]):.2f}), "
                # f"z({np.min(points[:,2]):.2f}-{np.max(points[:,2]):.2f})")
            combined_transformation = rotation @ np.linalg.inv(cam_intrinsics)
            points = (combined_transformation @ points.T).T
            points += translation
            # print(f"转换后点数: {len(points)}")
            # print(f"转换后点云范围: x({np.min(points[:,0]):.2f}-{np.max(points[:,0]):.2f}), "
            #     f"y({np.min(points[:,1]):.2f}-{np.max(points[:,1]):.2f}), "
            #     f"z({np.min(points[:,2]):.2f}-{np.max(points[:,2]):.2f})")
            dx = np.asarray([row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]])
            cx = np.asarray([np.round(row[1]/row[2] - 0.5) for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)
            nx = np.asarray([(row[1] - row[0]) / row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)

            grid = np.zeros((nx[0], nx[1], nx[2]))
            idx_lidar = np.round(np.array([cx]) - points/dx - 0.5).astype(int)
            idx_lidar = idx_lidar[
                (idx_lidar[:,0] >= 0) * (idx_lidar[:,0] < nx[0]) * \
                (idx_lidar[:,1] >= 0) * (idx_lidar[:,1] < nx[1]) * \
                (idx_lidar[:,2] >= 0) * (idx_lidar[:,2] < nx[2])
            ]
            grid[idx_lidar[:,0], idx_lidar[:,1], idx_lidar[:,2]] = 1
            grid = grid.transpose((2,0,1))
            temporal_grid.append(grid)
        
        # print(f"点云非零元素数量: {np.count_nonzero(temporal_grid)}")
        # print(f"点云平均值: {np.mean(temporal_grid)}")

        return np.stack(temporal_grid)