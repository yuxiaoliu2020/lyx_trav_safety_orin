#! /home/liu/miniforge3/envs/lyx/bin/python
import rospy
import numpy as np
import cupy as cp
import math
import tf
import time
import os
import sys
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from std_srvs.srv import Empty, EmptyResponse
from nav_msgs.msg import OccupancyGrid, Odometry
from trav_safety.msg import TraversabilityMap, SafetyMap
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# from lyx_mppi_cupy import MPPI_CuPy, Config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../nav_realcar'))
# from src.lyx_mppi_nmpc import MPPI_CuPy, Config
from src.lyx_mppi_cupy import MPPI_CuPy, Config


class MPPINavigationNode:
    """实现基于MPPI的多路点导航的ROS节点（使用全局polyline+局部前瞻目标）"""
    
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('mppi_navigation_node')
        
        # 加载参数
        self.load_params()
        
        # 初始化状态变量
        self.current_pose = None
        self.current_yaw = 0.0
        self.safety_map = None
        self.traction_map = None
        self.map_info = None
        self.traction_map_info = None
        self.goal_reached = False
        self.stopped = False

        # === map_lock: 局部地图锁定参考（用于把世界目标投到“当前地图的局部坐标系”）===
        # 这些变量会在 trav_map_callback 中首次收到地图时被刷新。
        self.map_lock_valid = False
        self.map_lock_pose = None
        self.map_lock_yaw = 0.0
        self.map_lock_stamp = None
        self.map_lock_seq = 0

        # 地图更新标记：回调里置 True，必要时在控制循环里兜底更新一次
        self._maps_dirty = False

        # 目标点队列管理
        self.waypoints = []  # 存储所有目标点 (在 load_params 中填充)
        self.current_waypoint_index = 0  # 当前目标点索引
        self.all_goals_reached = False  # 所有目标点是否已到达

        # 全局路径（polyline），在 world/map/ENU 坐标系中
        # 格式: list[(x, y)]，在 start_navigation 时用当前位置+所有waypoints构造
        self.global_path = []
        self.global_path_cumlen = []  # 与global_path等长的累计弧长数组
        self.lookahead_dist = rospy.get_param('~lookahead_dist', 5.0)     # 沿polyline前瞻距离（米）
        self.local_target_max_radius = 5.0     # base_link前方5m半圆半径
        
        # 导航控制状态
        self.navigation_active = rospy.get_param('~auto_start', False)

        # 添加导航时间记录变量
        self.navigation_start_time = None
        self.navigation_end_time = None
        self.navigation_total_time = 0.0

        # 添加路径长度记录变量
        self.total_path_length = 0.0
        self.last_position = None
        self.is_recording_path = False

        # 世界坐标系下的实际轨迹记录（Path）
        # 轨迹点列表，在position_callback中追加
        self.world_traj = []
        # 轨迹采样间隔（米），避免频率太高点太密；可通过参数~traj_min_dist配置
        self.traj_min_dist = rospy.get_param('~traj_min_dist', 0.05)
        # 最近一个加入轨迹的点（世界坐标系）
        self.last_traj_x = None
        self.last_traj_y = None
        
        # 创建启动服务
        rospy.Service('start_navigation', Empty, self.start_navigation_service)
        
        # 创建发布者
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.mppi_path_pub = rospy.Publisher('/mppi_path', Path, queue_size=1)
        self.world_traj_pub = rospy.Publisher('/mppi_world_traj', Path, queue_size=1)
        self.pose_vis_pub = rospy.Publisher('/mppi_pose_vis', PoseStamped, queue_size=1)
        # 新增：补偿后虚拟位姿轨迹
        self.comp_traj_pub = rospy.Publisher('/mppi_comp_traj', Path, queue_size=1)
        self.waypoints_marker_pub = rospy.Publisher('/mppi_waypoints', MarkerArray, queue_size=1)
        
        # 订阅必要话题
        rospy.Subscriber(self.pose_topic, Odometry, self.position_callback)
        rospy.Subscriber(self.safety_map_topic, SafetyMap, self.safety_map_callback)
        rospy.Subscriber(self.trav_map_topic, TraversabilityMap, self.trav_map_callback)
        
        # 创建MPPI控制器实例
        self.setup_mppi_controller()
        
        # 启动定时循环
        rospy.Timer(rospy.Duration(1.0/self.control_rate), self.control_loop)
        
        if self.navigation_active:
            rospy.loginfo("MPPI导航节点初始化完成，自动启动模式，等待传感器数据...")
        else:
            rospy.loginfo("MPPI导航节点初始化完成，等待启动命令 (rosservice call /start_navigation)...")

    def build_global_path(self, start_x, start_y):
        """根据当前起点和所有路点构建全局polyline路径
        路径坐标全部在世界/ENU/map坐标系下
        """
        if len(self.waypoints) == 0:
            rospy.logwarn("尚未配置任何路点，无法构建全局路径")
            self.global_path = []
            self.global_path_cumlen = []
            return

        # 起点 + 所有路点
        pts = [(start_x, start_y)] + self.waypoints
        path = []
        cumlen = []
        total = 0.0
        prev = pts[0]
        path.append(prev)
        cumlen.append(0.0)
        for p in pts[1:]:
            dx = p[0] - prev[0]
            dy = p[1] - prev[1]
            seg_len = math.hypot(dx, dy)
            if seg_len <= 1e-6:
                # 非常近的点直接跳过，避免数值问题
                continue
            total += seg_len
            path.append(p)
            cumlen.append(total)
            prev = p
        self.global_path = path
        self.global_path_cumlen = cumlen
        rospy.loginfo(f"全局路径已构建，共 {len(self.global_path)} 个顶点，总长约 {total:.2f} m")

    def find_nearest_point_on_path(self, x, y):
        """在 global_path 上找到到点(x,y)最近的点及其弧长位置
        返回: (near_x, near_y, s_near)
        若global_path为空则返回(None, None, None)
        """
        if not self.global_path or len(self.global_path) < 2:
            return None, None, None

        best_dist2 = float('inf')
        best_pt = None
        best_s = 0.0

        for i in range(len(self.global_path) - 1):
            x0, y0 = self.global_path[i]
            x1, y1 = self.global_path[i + 1]
            seg_dx = x1 - x0
            seg_dy = y1 - y0
            seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len2 <= 1e-9:
                continue
            # 投影比例 t ∈ [0,1]
            t = ((x - x0) * seg_dx + (y - y0) * seg_dy) / seg_len2
            t = max(0.0, min(1.0, t))
            proj_x = x0 + t * seg_dx
            proj_y = y0 + t * seg_dy
            dx = x - proj_x
            dy = y - proj_y
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_pt = (proj_x, proj_y)
                seg_len = math.sqrt(seg_len2)
                s0 = self.global_path_cumlen[i]
                best_s = s0 + t * seg_len

        if best_pt is None:
            return None, None, None
        return best_pt[0], best_pt[1], best_s

    def sample_path_at_s(self, s):
        """在弧长s处采样global_path上的点
        要求 global_path 和 global_path_cumlen 已经构建好
        超出范围则饱和在路径两端
        返回 (x, y)
        """
        if not self.global_path:
            return None, None
        if len(self.global_path) == 1:
            return self.global_path[0]

        # 若s超出范围，直接返回端点
        if s <= 0.0:
            return self.global_path[0]
        if s >= self.global_path_cumlen[-1]:
            return self.global_path[-1]

        # 在累积弧长数组中查找 s 所在的线段
        for i in range(len(self.global_path_cumlen) - 1):
            s0 = self.global_path_cumlen[i]
            s1 = self.global_path_cumlen[i + 1]
            if s0 <= s <= s1:
                ratio = (s - s0) / max(1e-9, (s1 - s0))
                x0, y0 = self.global_path[i]
                x1, y1 = self.global_path[i + 1]
                x = x0 + ratio * (x1 - x0)
                y = y0 + ratio * (y1 - y0)
                return x, y

        # 理论上不会到这里，兜底
        return self.global_path[-1]

    def _pose_yaw_from_pose(self, pose):
        """从geometry_msgs/Pose中提取yaw"""
        q = pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
        return yaw

    def _transform_world_to_lock(self, wx: float, wy: float):
        """世界系点 -> map_lock系 (锁定时刻base_link)"""
        if not self.map_lock_valid or self.map_lock_pose is None:
            return None, None
        ox = self.map_lock_pose.position.x
        oy = self.map_lock_pose.position.y
        yaw0 = self.map_lock_yaw
        dx = wx - ox
        dy = wy - oy
        lx = dx * math.cos(-yaw0) - dy * math.sin(-yaw0)
        ly = dx * math.sin(-yaw0) + dy * math.cos(-yaw0)
        return lx, ly

    def _transform_lock_to_world(self, lx: float, ly: float):
        """map_lock系点 -> 世界系"""
        if not self.map_lock_valid or self.map_lock_pose is None:
            return None, None
        ox = self.map_lock_pose.position.x
        oy = self.map_lock_pose.position.y
        yaw0 = self.map_lock_yaw
        wx = ox + (lx * math.cos(yaw0) - ly * math.sin(yaw0))
        wy = oy + (lx * math.sin(yaw0) + ly * math.cos(yaw0))
        return wx, wy

    def _current_state_in_lock(self):
        """返回当前机器人在map_lock系下的(x,y,yaw)"""
        if self.current_pose is None or not self.map_lock_valid or self.map_lock_pose is None:
            return None
        cx = self.current_pose.position.x
        cy = self.current_pose.position.y
        cyaw = self.current_yaw

        dx = cx - self.map_lock_pose.position.x
        dy = cy - self.map_lock_pose.position.y

        x_l = dx * math.cos(-self.map_lock_yaw) - dy * math.sin(-self.map_lock_yaw)
        y_l = dx * math.sin(-self.map_lock_yaw) + dy * math.cos(-self.map_lock_yaw)
        yaw_l = cyaw - self.map_lock_yaw
        # wrap to [-pi, pi]
        yaw_l = math.atan2(math.sin(yaw_l), math.cos(yaw_l))
        return (x_l, y_l, yaw_l)

    def _refresh_map_lock_if_possible(self, stamp=None):
        """在收到新地图时刷新map_lock（需要已有里程计位姿）"""
        if self.current_pose is None:
            return
        self.map_lock_pose = self.current_pose
        self.map_lock_yaw = self.current_yaw
        self.map_lock_stamp = stamp if stamp is not None else rospy.Time.now()
        self.map_lock_valid = True
        self.map_lock_seq += 1

    def compute_lookahead_target_world(self, pose=None):
        """根据给定世界位姿和global_path计算世界坐标系下的前瞻目标P_lookahead。

        pose=None则使用当前里程计位姿。
        """
        if pose is None:
            pose = self.current_pose
        if pose is None or not self.global_path:
            return None, None

        robot_x = pose.position.x
        robot_y = pose.position.y

        near_x, near_y, s_near = self.find_nearest_point_on_path(robot_x, robot_y)
        if near_x is None:
            return None, None

        s_lookahead = s_near + self.lookahead_dist
        look_x, look_y = self.sample_path_at_s(s_lookahead)
        return look_x, look_y

    def get_target_point(self, ref_pose=None, ref_yaw=None):
        """获取“局部地图坐标系(map_lock)”下的目标点。

        - ref_pose/ref_yaw: 作为局部地图坐标系原点的世界位姿（默认使用map_lock）。
        - 返回的(local_x, local_y)位于 map_lock 坐标系。
        """
        if ref_pose is None:
            if not self.map_lock_valid or self.map_lock_pose is None:
                return None, None
            ref_pose = self.map_lock_pose
        if ref_yaw is None:
            ref_yaw = self.map_lock_yaw if self.map_lock_valid else 0.0

        # 读取调试开关（动态参数）
        use_front_test = rospy.get_param('~use_front_test_waypoint', False)

        if use_front_test:
            # 以ref_pose/ref_yaw为“世界系下的正前方3m”（用于测试lock系）
            x0 = ref_pose.position.x
            y0 = ref_pose.position.y
            yaw = ref_yaw
            world_goal_x = x0 + 3.0 * math.cos(yaw)
            world_goal_y = y0 + 3.0 * math.sin(yaw)
            goal_desc = "FRONT_TEST(3m,LOCK_REF)"
        else:
            look_x, look_y = self.compute_lookahead_target_world(pose=self.current_pose)
            if look_x is None:
                if len(self.waypoints) == 0:
                    return None, None
                world_goal_x, world_goal_y = self.waypoints[self.current_waypoint_index]
                goal_desc = f"WAYPOINT_FALLBACK[{self.current_waypoint_index}]"
            else:
                world_goal_x, world_goal_y = look_x, look_y
                goal_desc = "GLOBAL_LOOKAHEAD"

        # 世界系 -> map_lock系
        dx = world_goal_x - ref_pose.position.x
        dy = world_goal_y - ref_pose.position.y
        local_x = dx * math.cos(-ref_yaw) - dy * math.sin(-ref_yaw)
        local_y = dx * math.sin(-ref_yaw) + dy * math.cos(-ref_yaw)

        rospy.loginfo(
            f"[LOCAL_GOAL_LOCK] mode={goal_desc}, lock_seq={self.map_lock_seq}, "
            f"lock_world=({ref_pose.position.x:.2f},{ref_pose.position.y:.2f}), lock_yaw={ref_yaw:.2f} rad, "
            f"world_goal=({world_goal_x:.2f},{world_goal_y:.2f}), local_goal_lock=({local_x:.2f},{local_y:.2f})"
        )

        return local_x, local_y

    def start_navigation_service(self, req):
        """启动导航的服务回调函数"""
        if not self.navigation_active:
            self.load_params()
            if self.current_pose is None:
                rospy.logwarn("无法启动导航：未收到/fixposition/odometry_enu话题消息")
                return EmptyResponse()
                
            if self.safety_map is None or self.traction_map is None:
                rospy.logwarn("无法启动导航：未收到地图数据，请确保安全地图和通过性地图正常发布")
                return EmptyResponse()
            
            self.navigation_active = True
            self.publish_cmd_vel(0.2, 0.0)

            self.navigation_start_time = rospy.Time.now()
            rospy.loginfo(f"开始记录导航时间: {self.navigation_start_time.to_sec()}")

            self.total_path_length = 0.0
            self.last_position = None
            self.is_recording_path = True
            
            initial_x = self.current_pose.position.x
            initial_y = self.current_pose.position.y

            self.build_global_path(initial_x, initial_y)

            # 在世界坐标系下发布一次所有路点的Marker，方便在RViz中检查
            frame_id = self.pose_topic  # 实际上 /fixposition/odometry_enu 的 frame
            # 更稳妥地用最近一次odom的frame_id
            # 这里复用world_traj中最后一个点的frame_id，如果有的话
            if self.world_traj:
                frame_id = self.world_traj[-1].header.frame_id
            else:
                frame_id = "world"
            self.publish_waypoints_markers(frame_id)

            current_goal_num = self.current_waypoint_index + 1
            total_goals = len(self.waypoints)
            rospy.loginfo(f"导航开始！当前位置: ({initial_x:.2f}, {initial_y:.2f})")
            rospy.loginfo(f"将按顺序导航至 {total_goals} 个目标点")
            for i, (x, y) in enumerate(self.waypoints):
                dx = x - initial_x
                dy = y - initial_y
                dist = math.sqrt(dx*dx + dy*dy)
                rospy.loginfo(f"  目标点 {i+1}: ({x:.2f}, {y:.2f}) - 距离当前位置: {dist:.2f}米")
            rospy.loginfo(f"当前导航至目标点 {current_goal_num}/{total_goals}: ({self.goal_x:.2f}, {self.goal_y:.2f})")
            
            # 重置状态确保导航可以正常开始
            self.goal_reached = False
            self.all_goals_reached = False
            self.stopped = False
        else:
            rospy.loginfo("导航已经处于激活状态")
        return EmptyResponse()

    def load_params(self):
        """加载ROS参数"""
        self.safety_map_topic = rospy.get_param('~safety_map_topic', '/safety_map')
        self.trav_map_topic = rospy.get_param('~traversability_map_topic', '/traversability_map')
        self.pose_topic = rospy.get_param('~pose_topic', '/fixposition/odometry_enu')
        self.control_rate = rospy.get_param('~control_rate', 10.0)  # Hz
        self.goal_threshold = rospy.get_param('~goal_threshold', 1.0)  # 米
        
        # 加载目标点数组参数
        waypoints_x = rospy.get_param('~waypoints_x', [100.0])
        waypoints_y = rospy.get_param('~waypoints_y', [100.0])
        
        # 确保x和y坐标数组长度一致
        if len(waypoints_x) != len(waypoints_y):
            rospy.logwarn("目标点x和y坐标数量不匹配，使用较短的数组长度")
            min_len = min(len(waypoints_x), len(waypoints_y))
            waypoints_x = waypoints_x[:min_len]
            waypoints_y = waypoints_y[:min_len]
        
        self.waypoints = [(x, y) for x, y in zip(waypoints_x, waypoints_y)]
        self.current_waypoint_index = 0
        self.goal_x, self.goal_y = self.waypoints[0]
        rospy.loginfo(f"加载了 {len(self.waypoints)} 个目标点")
        for i, (x, y) in enumerate(self.waypoints):
            rospy.loginfo(f"  目标点 {i+1}: ({x:.2f}, {y:.2f})")
        
        # 配置参数
        self.mppi_horizon = rospy.get_param('~mppi_horizon', 100)
        self.mppi_dt = rospy.get_param('~mppi_dt', 0.1)
        self.mppi_num_samples = rospy.get_param('~mppi_num_samples', 1024)
        self.mppi_iterations = rospy.get_param('~mppi_iterations', 8)

        # === 控制噪声 ===
        self.noise_sigma_v = rospy.get_param('~noise_sigma_v', 0.30)
        self.noise_sigma_w = rospy.get_param('~noise_sigma_w', 0.10)

        # === 控制边界 ===
        self.v_min = rospy.get_param('~v_min', 0.1)
        self.v_max = rospy.get_param('~v_max', 0.5)
        self.w_min = rospy.get_param('~w_min', -0.5)
        self.w_max = rospy.get_param('~w_max', 0.5)

        # === 单步增量限制（Δu）===
        self.max_delta_v = rospy.get_param('~max_delta_v', 0.2)
        self.max_delta_w = rospy.get_param('~max_delta_w', 0.4)

        # MPPI 温度
        self.lambda_ = rospy.get_param('~lambda', 0.1)

        # 地图参数（与 trav_safety 输出对齐）
        self.map_resolution = rospy.get_param('~map_resolution', 0.1)
        self.map_origin_x = rospy.get_param('~map_origin_x', -2.0)
        self.map_origin_y = rospy.get_param('~map_origin_y', -5.0)

        # NMPC 二次型权重
        self.q_pos = rospy.get_param('~q_pos', 1.0)
        self.q_theta = rospy.get_param('~q_theta', 0.1)
        self.qN_pos = rospy.get_param('~qN_pos', 2.0)
        self.qN_theta = rospy.get_param('~qN_theta', 0.2)
        self.r_v = rospy.get_param('~r_v', 0.01)
        self.r_w = rospy.get_param('~r_w', 0.01)

        # μ/ν 与 safety 权重
        self.w_mu = rospy.get_param('~w_mu', 0.5)
        self.w_nu = rospy.get_param('~w_nu', 0.5)
        self.w_safety = rospy.get_param('~w_safety', 0.5)

    def setup_mppi_controller(self):
        """根据当前参数和地图，创建或更新 MPPI 控制器实例。"""
        if self.traction_map is None or self.safety_map is None:
            rospy.logwarn("MPPI 控制器等待地图数据 (traction_map / safety_map 为空)")
            self.mppi = None
            return

        cfg = Config(
            horizon_MPPI=self.mppi_horizon,
            dt=self.mppi_dt,
            num_samples=self.mppi_num_samples,
            num_iterations=self.mppi_iterations,
            noise_sigma_v=self.noise_sigma_v,
            noise_sigma_w=self.noise_sigma_w,
            v_min=self.v_min,
            v_max=self.v_max,
            w_min=self.w_min,
            w_max=self.w_max,
            max_delta_v=self.max_delta_v,
            max_delta_w=self.max_delta_w,
            lambda_=self.lambda_,
            map_resolution=self.map_resolution,
            map_origin_x=self.map_origin_x,
            map_origin_y=self.map_origin_y,
            q_pos=self.q_pos,
            q_theta=self.q_theta,
            qN_pos=self.qN_pos,
            qN_theta=self.qN_theta,
            r_v=self.r_v,
            r_w=self.r_w,
            w_mu=self.w_mu,
            w_nu=self.w_nu,
            w_safety=self.w_safety,
        )

        try:
            self.mppi = MPPI_CuPy(self.traction_map, self.safety_map, cfg=cfg)
            rospy.loginfo("MPPI_CuPy 控制器初始化完成")
        except Exception as e:
            rospy.logerr(f"初始化 MPPI_CuPy 失败: {e}")
            self.mppi = None

    def safety_map_callback(self, msg):
        """处理安全地图消息回调"""
        height = msg.height
        width = msg.width

        safety_data = np.array(msg.safety_data, dtype=np.float32).reshape(height, width)
        # safety_data = np.ones((height, width), dtype=np.float32)
        self.safety_map = safety_data

        self.map_info = {
            'width': width,
            'height': height,
            'resolution': msg.resolution,
            'origin_x': msg.origin.position.x,
            'origin_y': msg.origin.position.y,
            'origin_orientation_x': msg.origin.orientation.x,
            'origin_orientation_y': msg.origin.orientation.y,
            'origin_orientation_z': msg.origin.orientation.z,
            'origin_orientation_w': msg.origin.orientation.w
        }

        # 标记地图已更新，触发GPU侧更新（若控制器已就绪）
        self._maps_dirty = True
        if getattr(self, 'mppi', None) is not None:
            self.update_mppi_maps()
            self._maps_dirty = False

    def trav_map_callback(self, msg):
        """处理通过性地图消息回调"""
        height = msg.height
        width = msg.width

        mu_data = np.array(msg.mu_data, dtype=np.float32).reshape(height, width)
        nu_data = np.array(msg.nu_data, dtype=np.float32).reshape(height, width)

        self.traction_map = np.stack([mu_data, nu_data], axis=2)
        self.traction_map_info = {
            'width': width,
            'height': height,
            'resolution': msg.resolution,
            'origin_x': msg.origin.position.x,
            'origin_y': msg.origin.position.y
        }

        # 关键：每次收到新地图时，把“局部地图坐标系原点”锁定在此刻里程计位姿
        self._refresh_map_lock_if_possible(stamp=msg.header.stamp)

        # 标记地图已更新，触发GPU侧更新（若控制器已就绪）
        self._maps_dirty = True
        if getattr(self, 'mppi', None) is not None:
            self.update_mppi_maps()
            self._maps_dirty = False

    def position_callback(self, msg):
        """位姿回调：更新当前位姿和航向角（来自 /fixposition/odometry_enu）"""
        self.current_pose = msg.pose.pose
        orientation = msg.pose.pose.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        # tf.transformations.euler_from_quaternion 返回 (roll, pitch, yaw)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        self.current_yaw = yaw

        # 记录并发布世界坐标系下的实际轨迹
        frame_id = msg.header.frame_id if msg.header.frame_id else "world"

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 基于距离的采样，避免在静止或移动很小时时疯狂刷点
        need_append = False
        if self.last_traj_x is None:
            need_append = True
        else:
            dx = x - self.last_traj_x
            dy = y - self.last_traj_y
            dist = math.hypot(dx, dy)
            if dist >= self.traj_min_dist:
                need_append = True

        if need_append:
            self.last_traj_x = x
            self.last_traj_y = y

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = msg.header.stamp
            pose_stamped.header.frame_id = frame_id
            pose_stamped.pose = msg.pose.pose

            self.world_traj.append(pose_stamped)

            path_msg = Path()
            path_msg.header.stamp = msg.header.stamp
            path_msg.header.frame_id = frame_id
            path_msg.poses = self.world_traj

            self.world_traj_pub.publish(path_msg)

        # 直接转发当前Pose用于RViz显示（箭头）
        pose_vis = PoseStamped()
        pose_vis.header.stamp = msg.header.stamp
        pose_vis.header.frame_id = frame_id
        pose_vis.pose = msg.pose.pose
        self.pose_vis_pub.publish(pose_vis)

        return

    def publish_waypoints_markers(self, frame_id):
        """在世界坐标系下用MarkerArray可视化所有waypoints"""
        if not self.waypoints:
            rospy.logwarn("[WAYPOINT_MARKER] 没有任何waypoints，跳过可视化")
            return

        if not self.global_path:
            rospy.logwarn("[WAYPOINT_MARKER] global_path为空，可能尚未调用build_global_path")

        rospy.loginfo(f"[WAYPOINT_MARKER] 使用frame_id={frame_id}, waypoints={len(self.waypoints)}, global_path顶点数={len(self.global_path)}")

        markers = MarkerArray()
        now = rospy.Time.now()

        # 1) 用一个线条把所有路点连起来
        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = now
        line.ns = "mppi_waypoints"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.2  # 线宽
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 1.0
        line.pose.orientation.w = 1.0

        # 起点：当前build_global_path时的起点也在global_path[0]
        for x, y in self.global_path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            line.points.append(p)
        markers.markers.append(line)

        # 2) 每个路点画一个小球
        for i, (x, y) in enumerate(self.waypoints, start=1):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = now
            m.ns = "mppi_waypoints_points"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 0.5
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            markers.markers.append(m)

        self.waypoints_marker_pub.publish(markers)

    def update_mppi_maps(self):
        """在收到新的 traversability / safety 地图后更新 MPPI 控制器。"""
        # 当地图更新且已有控制器时，仅替换其内部地图，无需重建对象
        if hasattr(self, 'mppi') and self.mppi is not None:
            try:
                self.mppi.traction_map = cp.asarray(self.traction_map, dtype=cp.float32, order="C")
                self.mppi.safety_map = cp.asarray(self.safety_map, dtype=cp.float32, order="C")
                self.mppi.H, self.mppi.W = self.mppi.traction_map.shape[:2]
                self.mppi.traction_v = self.mppi.traction_map[:, :, 0].ravel()
                self.mppi.traction_w = self.mppi.traction_map[:, :, 1].ravel()
                self.mppi.safety_flat = self.mppi.safety_map.ravel()
            except Exception as e:
                rospy.logerr(f"更新 MPPI 地图失败，将尝试重新创建控制器: {e}")
                self.setup_mppi_controller()
        else:
            # 尚未创建控制器，尝试创建
            self.setup_mppi_controller()

    def control_loop(self, event):
        """主控制循环，以固定频率执行"""
        if not self.navigation_active:
            return

        if self.all_goals_reached or self.stopped:
            return
            
        if not self.current_pose or self.safety_map is None or self.traction_map is None:
            rospy.logwarn_throttle(5.0, "等待数据就绪：位姿、安全地图或通过性地图")
            return

        if not self.map_lock_valid:
            rospy.logwarn_throttle(2.0, "等待map_lock初始化：需要至少一帧通过性地图回调")
            return

        # 不在控制循环里更新GPU地图（避免高频拷贝）；
        # 若因启动时序导致控制器尚未初始化，则在这里兜底初始化一次。
        if getattr(self, 'mppi', None) is None:
            self.setup_mppi_controller()
            if getattr(self, 'mppi', None) is None:
                rospy.logwarn_throttle(2.0, "MPPI控制器尚未就绪，等待地图数据")
                return
        elif getattr(self, '_maps_dirty', False):
            # 兜底：如果有dirty但回调里没来得及更新（非常规情况），这里更新一次
            self.update_mppi_maps()
            self._maps_dirty = False

        try:
            # 目标：固定在map_lock坐标系下
            target_x, target_y = self.get_target_point(ref_pose=self.map_lock_pose, ref_yaw=self.map_lock_yaw)
            if target_x is None or target_y is None:
                rospy.logwarn_throttle(5.0, "无法确定局部目标点")
                return

            # 当前状态：在map_lock坐标系下（地图两帧之间持续变化）
            state0 = self._current_state_in_lock()
            if state0 is None:
                rospy.logwarn_throttle(5.0, "无法计算当前state0(map_lock系)")
                return

            goal_state = (target_x, target_y)

            current_goal_num = self.current_waypoint_index + 1
            total_goals = len(self.waypoints)
            rospy.loginfo(
                f"[MPPI_LOCK] goal {current_goal_num}/{total_goals}, lock_seq={self.map_lock_seq}, "
                f"state0_lock=({state0[0]:.2f},{state0[1]:.2f},{state0[2]:.2f}), goal_lock=({target_x:.2f},{target_y:.2f})"
            )

            control_sequence = self.mppi.solve(state0, goal_state)

            v = float(control_sequence[0, 0])
            w = float(control_sequence[0, 1])

            rospy.loginfo(f"控制命令: v={v:.2f} m/s, w={w:.2f} rad/s")

            if self.goal_reached or self.stopped:
                self.publish_cmd_vel(0.0, 0.0)
                return

            self.publish_cmd_vel(v, w)

            self.visualize_mppi_path(goal_state)

            self.mppi.shift_and_update()

        except Exception as e:
            rospy.logerr(f"MPPI计算错误: {e}")
            self.publish_cmd_vel(0.0, 0.0)

    def update_waypoint_progress(self):
        """根据当前位置和waypoints更新当前路点索引
        - 必须进入每个路点1m范围内才算到达
        - 依次推进，全部完成后标记all_goals_reached
        """
        if self.current_pose is None or len(self.waypoints) == 0:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            self.all_goals_reached = True
            return

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        goal_x, goal_y = self.waypoints[self.current_waypoint_index]
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dist = math.hypot(dx, dy)

        if dist <= self.goal_threshold:
            rospy.loginfo(
                f"到达路点 {self.current_waypoint_index + 1}/{len(self.waypoints)}: ({goal_x:.2f}, {goal_y:.2f}), 距离 {dist:.2f} m")
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.waypoints):
                self.all_goals_reached = True
                self.goal_reached = True
                self.stopped = True
                self.publish_cmd_vel(0.0, 0.0)
                rospy.loginfo("所有路点已完成导航，机器人停止。")

                self.navigation_end_time = rospy.Time.now()
                if self.navigation_start_time is not None:
                    self.navigation_total_time = (self.navigation_end_time - self.navigation_start_time).to_sec()
                    rospy.loginfo(f"本次导航总用时: {self.navigation_total_time:.2f} 秒")
                return
            else:
                self.goal_reached = False
                self.stopped = False
                self.goal_x, self.goal_y = self.waypoints[self.current_waypoint_index]
                rospy.loginfo(
                    f"切换到下一个路点 {self.current_waypoint_index + 1}/{len(self.waypoints)}: ({self.goal_x:.2f}, {self.goal_y:.2f})")

    def publish_cmd_vel(self, v, w):
        """发布速度控制命令"""
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_vel_pub.publish(cmd)
         
    def visualize_mppi_path(self, goal_state):
        """可视化MPPI采样路径"""
        if not hasattr(self.mppi, '_state_buf'):
            return

        try:
            # 在MPPI内部，状态仍然是以"当前时刻的base_link"为原点，
            # 因此这里依然从(0,0,0)开始即可。
            state0 = (0.0, 0.0, 0.0)
            trajectory, _ = self.mppi.get_state_rollout(state0, goal_state)

            path_msg = Path()
            path_msg.header.frame_id = "base_link"
            path_msg.header.stamp = rospy.Time.now()
            
            for i in range(len(trajectory)):
                x, y, _ = trajectory[i]
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)

            self.mppi_path_pub.publish(path_msg)
            
        except Exception as e:
            rospy.logwarn(f"无法可视化MPPI路径: {e}")
    
    def stop_robot(self):
        self.stopped = True
        self.goal_reached = True
        self.publish_cmd_vel(0.0, 0.0)
        
        rospy.loginfo("机器人已停止")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        nav_node = MPPINavigationNode()
        nav_node.run()
    except rospy.ROSInterruptException:
        pass