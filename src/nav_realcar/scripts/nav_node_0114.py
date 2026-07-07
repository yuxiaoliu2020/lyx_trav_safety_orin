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

        # === 延迟补偿相关 ===
        # 估计的里程计/RTK整体延迟（秒），由参数加载
        self.odom_delay = rospy.get_param('~odom_delay', 0.0)
        # 最近一次控制指令，用于做简单的前向积分
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_cmd_time = None

        # 补偿后虚拟位姿轨迹（Path），用于RViz对比
        self.comp_traj = []
        self.last_comp_traj_x = None
        self.last_comp_traj_y = None
        # 与实测轨迹使用相同的采样距离
        
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
    
    def predict_state(self, base_x, base_y, base_yaw, v, w, dt):
        """在平面上用与MPPI相同的一阶非完整车模型做前向积分，用于延迟补偿。

        参数:
            base_x, base_y, base_yaw: 当前(实际上是滞后)里程计位姿
            v, w: 最近一段时间的控制指令(假设在[0, dt]内基本不变)
            dt: 需要前向预测的时间(秒)，即估计的延迟
        返回:
            (x_pred, y_pred, yaw_pred)
        """
        if dt <= 0.0:
            return base_x, base_y, base_yaw

        # 使用与MPPI rollout中相同形式的一阶积分模型，但去掉通过性系数
        # MPPI中：
        #   x_{t+1} = x_t + v_eff * cos(th_t) * dt  (v_eff = v * traction_v)
        #   y_{t+1} = y_t + v_eff * sin(th_t) * dt
        #   th_{t+1} = th_t + w_eff * dt          (w_eff = w * traction_w)
        # 这里做延迟补偿时不考虑地形，通过性系数视为 1
        x_pred = base_x + v * math.cos(base_yaw) * dt
        y_pred = base_y + v * math.sin(base_yaw) * dt
        yaw_pred = base_yaw + w * dt

        # 将yaw归一化到[-pi, pi]
        yaw_pred = math.atan2(math.sin(yaw_pred), math.cos(yaw_pred))
        return x_pred, y_pred, yaw_pred

    def get_compensated_pose(self):
        """返回做了延迟补偿后的估计当前位姿。

        - 若 self.odom_delay <= 0，则直接返回 current_pose, current_yaw。
        - 否则，用最近一次控制命令(last_cmd_v, last_cmd_w)在平面上做dt秒的前向积分。
        """
        if self.current_pose is None:
            return None, None, None

        # 无延迟补偿直接返回
        if self.odom_delay <= 0.0:
            return (
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_yaw,
            )

        base_x = self.current_pose.position.x
        base_y = self.current_pose.position.y
        base_yaw = self.current_yaw

        # 使用参数设定的全局延迟做前向预测
        dt = self.odom_delay
        x_pred, y_pred, yaw_pred = self.predict_state(
            base_x, base_y, base_yaw,
            self.last_cmd_v, self.last_cmd_w,
            dt,
        )

        rospy.logdebug(
            "[DELAY_COMP] odom(x=%.2f,y=%.2f,yaw=%.2f), v=%.2f, w=%.2f, dt=%.2f -> pred(x=%.2f,y=%.2f,yaw=%.2f)",
            base_x, base_y, base_yaw,
            self.last_cmd_v, self.last_cmd_w,
            dt,
            x_pred, y_pred, yaw_pred,
        )

        return x_pred, y_pred, yaw_pred
         
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

    def compute_lookahead_target_world(self):
        """根据当前世界位姿和global_path计算世界坐标系下的前瞻目标P_lookahead"""
        if self.current_pose is None or not self.global_path:
            return None, None

        # 使用延迟补偿后的估计当前位姿
        robot_x, robot_y, _ = self.get_compensated_pose()
        if robot_x is None:
            return None

        # 1. 找到global_path上距离机器人最近的点 P_near 及其弧长 s_near
        near_x, near_y, s_near = self.find_nearest_point_on_path(robot_x, robot_y)
        if near_x is None:
            return None, None

        # 2. 从 s_near 沿路径前进 lookahead_dist 得到 s_lookahead
        s_lookahead = s_near + self.lookahead_dist
        look_x, look_y = self.sample_path_at_s(s_lookahead)

        rospy.logdebug(f"P_near = ({near_x:.2f}, {near_y:.2f}), P_lookahead = ({look_x:.2f}, {look_y:.2f}), s_near = {s_near:.2f}, s_lookahead = {s_lookahead:.2f}")
        return look_x, look_y

    def world_to_local(self, world_x, world_y):
        """将世界坐标转换为机器人局部坐标 (base_link)

        这里也使用经延迟补偿后的当前位姿，保持与轨迹规划一致。
        """
        if self.current_pose is None:
            return None, None

        robot_x, robot_y, robot_yaw = self.get_compensated_pose()
        if robot_x is None:
            return None, None

        dx = world_x - robot_x
        dy = world_y - robot_y

        local_x = dx * math.cos(-robot_yaw) - dy * math.sin(-robot_yaw)
        local_y = dx * math.sin(-robot_yaw) + dy * math.cos(-robot_yaw)
        
        return local_x, local_y
    
    def is_point_in_map_bounds(self, x, y):
        """检查给定的局部坐标点是否在地图范围内"""
        if self.safety_map is None or self.map_info is None:
            return False
        
        x_min = -2.0  # 后方2米
        x_max = 8.0   # 前方8米 
        y_min = -5.0  # 右侧5米
        y_max = 5.0   # 左侧5米
        
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)
    
    # def adjust_target_to_map_bounds(self, target_x, target_y):
    #     """如果目标点超出地图边界，将其调整到边界上 - 支持非对称布局"""
    #     if self.is_point_in_map_bounds(target_x, target_y):
    #         return target_x, target_y
        
    #     if self.map_info is None:
    #         return 0, 0  # 没有地图信息时，默认为原点
        
    #     x_min = -2.0  # 后方2米
    #     x_max = 8.0   # 前方8米
    #     y_min = -5.0  # 右侧5米
    #     y_max = 5.0   # 左侧5米
        
    #     # 寻找射线与矩形边界的交点, 使用参数方程
    #     t_values = []
        
    #     # 与x_min边界相交
    #     if target_x != 0:
    #         t_x_min = (x_min - 0) / target_x
    #         if t_x_min > 0:
    #             y_at_x_min = t_x_min * target_y
    #             if y_min <= y_at_x_min <= y_max:
    #                 t_values.append((t_x_min, x_min, y_at_x_min))
        
    #     # 与x_max边界相交
    #     if target_x != 0:
    #         t_x_max = (x_max - 0) / target_x
    #         if t_x_max > 0:
    #             y_at_x_max = t_x_max * target_y
    #             if y_min <= y_at_x_max <= y_max:
    #                 t_values.append((t_x_max, x_max, y_at_x_max))
        
    #     # 与y_min边界相交
    #     if target_y != 0:
    #         t_y_min = (y_min - 0) / target_y
    #         if t_y_min > 0:
    #             x_at_y_min = t_y_min * target_x
    #             if x_min <= x_at_y_min <= x_max:
    #                 t_values.append((t_y_min, x_at_y_min, y_min))
        
    #     # 与y_max边界相交
    #     if target_y != 0:
    #         t_y_max = (y_max - 0) / target_y
    #         if t_y_max > 0:
    #             x_at_y_max = t_y_max * target_x
    #             if x_min <= x_at_y_max <= x_max:
    #                 t_values.append((t_y_max, x_at_y_max, y_max))
        
    #     # 选择最近的交点
    #     if t_values:
    #         t_values.sort()  # 按t值排序，最小的t对应最近的交点
    #         _, bound_x, bound_y = t_values[0]
    #     else:
    #         # 如果没有找到交点，使用对角点
    #         if target_x > 0:
    #             bound_x = x_max
    #         else:
    #             bound_x = x_min
            
    #         if target_y > 0:
    #             bound_y = y_max
    #         else:
    #             bound_y = y_min
        
    #     rospy.loginfo(f"目标点 ({target_x:.2f}, {target_y:.2f}) 超出地图范围，调整为 ({bound_x:.2f}, {bound_y:.2f})")
    #     return bound_x, bound_y

    def limit_to_front_semicircle(self, x, y):
        """将base_link系下的目标点限制在车辆前方local_target_max_radius半径的半圆内
        条件: x >= 0 且 x^2 + y^2 <= R^2
        若目标在后方或超出半径，则投影到前方半圆边界
        """
        R = self.local_target_max_radius
        r = math.hypot(x, y)
        if r < 1e-6:
            # 正前方很近的一个点，直接放在前方极小距离
            return 0.1, 0.0

        # 若在半圆内且在前方，则直接返回
        if x >= 0.0 and r <= R:
            return x, y

        # 将点投影到前方半圆边界上
        # 先把向量单位化，再乘以半径R，并强制x>=0
        ux = x / r
        uy = y / r
        # 如果单位向量指向后方，则改为正前方方向
        if ux < 0.0:
            ux = -ux
            uy = -uy
        new_x = ux * R
        new_y = uy * R
        return new_x, new_y
    
    def get_target_point(self):
        """获取局部坐标下的目标点

        默认行为：
        - 使用全局polyline(global_path) + 当前位姿计算前瞻目标点；
        - 将前瞻点转换到 base_link 坐标系；
        - 再将其截断到局部地图矩形范围以及车辆前方半圆边界上；
        - 作为 MPPI 的局部目标。

        调试模式（~use_front_test_waypoint=True）：
        - 忽略 global_path，使用“世界系下正前方3m”作为测试点；
        """
        if self.current_pose is None:
            return None, None

        # 读取调试开关（动态参数，方便运行时切换）
        use_front_test = rospy.get_param('~use_front_test_waypoint', False)

        if use_front_test:
            # 调试模式：世界系下正前方3m
            x0 = self.current_pose.position.x
            y0 = self.current_pose.position.y
            yaw = self.current_yaw

            world_goal_x = x0 + 3.0 * math.cos(yaw)
            world_goal_y = y0 + 3.0 * math.sin(yaw)
            goal_desc = "FRONT_TEST(3m)"
        else:
            # 正常模式：基于全局polyline做前瞻
            look_x, look_y = self.compute_lookahead_target_world()
            if look_x is None:
                # 若global_path尚未构建或异常，则退化为当前路点
                if len(self.waypoints) == 0:
                    return None, None
                world_goal_x, world_goal_y = self.waypoints[self.current_waypoint_index]
                goal_desc = f"WAYPOINT_FALLBACK[{self.current_waypoint_index}]"
            else:
                world_goal_x, world_goal_y = look_x, look_y
                goal_desc = "GLOBAL_LOOKAHEAD"

        # 1) 世界系 → base_link 系
        local_x, local_y = self.world_to_local(world_goal_x, world_goal_y)
        if local_x is None:
            return None, None

        # 2) 先裁剪到局部地图矩形范围
        # local_x, local_y = self.adjust_target_to_map_bounds(local_x, local_y)

        # 3) 再裁剪到车辆前方半圆边界
        local_x, local_y = self.limit_to_front_semicircle(local_x, local_y)

        # re_world_x = self.current_pose.position.x + (
        # local_x * math.cos(self.current_yaw) - local_y * math.sin(self.current_yaw)
        # )
        # re_world_y = self.current_pose.position.y + (
        #     local_x * math.sin(self.current_yaw) + local_y * math.cos(self.current_yaw)
        # )
        # rospy.loginfo(f"[CHECK] world_goal=({world_goal_x:.2f},{world_goal_y:.2f}), "
        #             f"re_world=({re_world_x:.2f},{re_world_y:.2f})")

        # 打印调试信息：世界坐标、局部坐标、当前yaw
        rospy.loginfo(
            f"[LOCAL_GOAL] mode={goal_desc}, "
            f"world_goal=({world_goal_x:.2f}, {world_goal_y:.2f}), "
            f"odom_raw=({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
            f"odom_yaw_raw={self.current_yaw:.2f} rad, "
            f"local_goal=({local_x:.2f}, {local_y:.2f})")

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
        
        # 构建目标点列表
        self.waypoints = [(x, y) for x, y in zip(waypoints_x, waypoints_y)]
        
        # 设置当前目标
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

        # === 代价权重 ===
        self.dist_weight = rospy.get_param('~dist_weight', 0.5)
        self.lambda_ = rospy.get_param('~lambda', 0.2)
        self.safety_weight = rospy.get_param('~safety_weight', 0.5)
        self.smooth_v_weight = rospy.get_param('~smooth_v_weight', 0.5)
        self.smooth_w_weight = rospy.get_param('~smooth_w_weight', 2.0)
        self.w_mu = rospy.get_param('~w_mu', 0.5)
        self.w_nu = rospy.get_param('~w_nu', 0.5)

        # NMPC权重矩阵参数
        self.nmpc_Q_x = rospy.get_param('~nmpc_Q_x', 1.0)
        self.nmpc_Q_y = rospy.get_param('~nmpc_Q_y', 1.0)
        self.nmpc_Q_th = rospy.get_param('~nmpc_Q_th', 0.1)
        self.nmpc_R_v = rospy.get_param('~nmpc_R_v', 0.1)
        self.nmpc_R_w = rospy.get_param('~nmpc_R_w', 0.1)
        self.nmpc_QN_x = rospy.get_param('~nmpc_QN_x', 10.0)
        self.nmpc_QN_y = rospy.get_param('~nmpc_QN_y', 10.0)
        self.nmpc_QN_th = rospy.get_param('~nmpc_QN_th', 1.0)

        # === 延迟补偿参数 ===
        # 估计的里程计延迟时间(秒)，例如1.0或2.0，可在launch中调参做sweep
        self.odom_delay = rospy.get_param('~odom_delay', 0.0)
        if self.odom_delay > 0.0:
            rospy.loginfo("[DELAY_COMP] 启用里程计延迟补偿, odom_delay = %.3f s", self.odom_delay)
        else:
            rospy.loginfo("[DELAY_COMP] 未启用里程计延迟补偿 (odom_delay <= 0)")

    def setup_mppi_controller(self):
        """初始化MPPI控制器"""
        traction_map_shape = (100, 100, 2)  # H, W, 2 (v, w)
        safety_map_shape = (100, 100)        # H, W

        default_traction_map = np.ones(traction_map_shape, dtype=np.float32)
        default_safety_map = np.ones(safety_map_shape, dtype=np.float32)

        mppi_config = Config()
        # 规划范围
        mppi_config.horizon_MPPI = self.mppi_horizon
        mppi_config.dt = self.mppi_dt
        mppi_config.num_samples = self.mppi_num_samples
        mppi_config.num_iterations = self.mppi_iterations

        # 控制噪声
        mppi_config.noise_sigma_v = self.noise_sigma_v
        mppi_config.noise_sigma_w = self.noise_sigma_w

        # 控制边界
        mppi_config.v_min = self.v_min
        mppi_config.v_max = self.v_max
        mppi_config.w_min = self.w_min
        mppi_config.w_max = self.w_max

        # 增量限幅
        mppi_config.max_delta_v = self.max_delta_v
        mppi_config.max_delta_w = self.max_delta_w

        # 成本函数权重
        mppi_config.dist_weight = self.dist_weight
        mppi_config.safety_weight = self.safety_weight
        mppi_config.lambda_ = self.lambda_
        mppi_config.smooth_v_weight = self.smooth_v_weight
        mppi_config.smooth_w_weight = self.smooth_w_weight
        mppi_config.w_mu = self.w_mu
        mppi_config.w_nu = self.w_nu

        # 地图分辨率（米/像素）和原点（保持写死或以后再暴露）
        mppi_config.map_resolution = 0.1
        mppi_config.map_origin_x = -2.0
        mppi_config.map_origin_y = -5.0

        # NMPC参数设置
        mppi_config.use_nmpc = self.use_nmpc
        mppi_config.Q_x = self.nmpc_Q_x
        mppi_config.Q_y = self.nmpc_Q_y
        mppi_config.Q_th = self.nmpc_Q_th
        mppi_config.R_v = self.nmpc_R_v
        mppi_config.R_w = self.nmpc_R_w
        mppi_config.QN_x = self.nmpc_QN_x
        mppi_config.QN_y = self.nmpc_QN_y
        mppi_config.QN_th = self.nmpc_QN_th

        # 创建MPPI控制器
        self.mppi = MPPI_CuPy(
            traction_map=default_traction_map, 
            safety_map=default_safety_map,
            cfg=mppi_config
        )
        
        # 初始化地图数据
        self.traction_map = default_traction_map
        self.safety_map = default_safety_map
        
        rospy.loginfo("MPPI+NMPC控制器初始化完成")
        if self.use_nmpc:
            rospy.loginfo("NMPC已启用，使用MPPI+NMPC级联控制")
        else:
            rospy.loginfo("NMPC已禁用，仅使用MPPI控制")

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
    
    def trav_map_callback(self, msg):
        """处理通过性地图消息回调"""
        height = msg.height
        width = msg.width

        mu_data = np.array(msg.mu_data, dtype=np.float32).reshape(height, width)
        nu_data = np.array(msg.nu_data, dtype=np.float32).reshape(height, width)

        # mu_data = np.ones((height, width), dtype=np.float32)
        # nu_data = np.ones((height, width), dtype=np.float32)
        
        self.traction_map = np.stack([mu_data, nu_data], axis=2)
        self.traction_map_info = {
            'width': width,
            'height': height,
            'resolution': msg.resolution,
            'origin_x': msg.origin.position.x,
            'origin_y': msg.origin.position.y
        }
    
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

        # === 额外：发布补偿后虚拟位姿轨迹，用于与实测轨迹对比 ===
        if self.current_pose is not None:
            comp_x, comp_y, comp_yaw = self.get_compensated_pose()
            if comp_x is not None:
                # 距离采样，避免点过密
                need_comp_append = False
                if self.last_comp_traj_x is None:
                    need_comp_append = True
                else:
                    dx_c = comp_x - self.last_comp_traj_x
                    dy_c = comp_y - self.last_comp_traj_y
                    dist_c = math.hypot(dx_c, dy_c)
                    if dist_c >= self.traj_min_dist:
                        need_comp_append = True

                if need_comp_append:
                    self.last_comp_traj_x = comp_x
                    self.last_comp_traj_y = comp_y

                    comp_pose = PoseStamped()
                    comp_pose.header.stamp = msg.header.stamp
                    comp_pose.header.frame_id = frame_id
                    comp_pose.pose.position.x = comp_x
                    comp_pose.pose.position.y = comp_y
                    comp_pose.pose.position.z = self.current_pose.position.z
                    comp_pose.pose.orientation = self.current_pose.orientation

                    self.comp_traj.append(comp_pose)

                    comp_path_msg = Path()
                    comp_path_msg.header.stamp = msg.header.stamp
                    comp_path_msg.header.frame_id = frame_id
                    comp_path_msg.poses = self.comp_traj

                    self.comp_traj_pub.publish(comp_path_msg)

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
        """更新MPPI控制器使用的地图"""
        if self.safety_map is not None and self.traction_map is not None:
            cp.copyto(self.mppi.safety_map, cp.asarray(self.safety_map, dtype=cp.float32))
            cp.copyto(self.mppi.traction_map, cp.asarray(self.traction_map, dtype=cp.float32))
            
            # 更新扁平化的地图（用于GPU上的快速访问）
            # self.mppi.risk_flat = self.mppi.risk_map.flatten()
            self.mppi.safety_flat = self.mppi.safety_map.flatten()
            self.mppi.traction_v = self.mppi.traction_map[:, :, 0].flatten()
            self.mppi.traction_w = self.mppi.traction_map[:, :, 1].flatten()
            self.mppi.H, self.mppi.W = self.mppi.traction_map.shape[:2]

            if self.map_info is not None:
                # self.mppi.cfg.map_origin_x = self.map_info['origin_x']
                # self.mppi.cfg.map_origin_y = self.map_info['origin_y']
                self.mppi.cfg.map_resolution = self.map_info['resolution']
            self.mppi.cfg.map_origin_x = -2.0
            self.mppi.cfg.map_origin_y = -5.0

    def control_loop(self, event):
        """主控制循环，以固定频率执行"""
        if not self.navigation_active:
            return

        if self.all_goals_reached or self.stopped:
            return
            
        if not self.current_pose or self.safety_map is None or self.traction_map is None:
            rospy.logwarn_throttle(5.0, "等待数据就绪：位姿、安全地图或通过性地图")
            return

        self.update_waypoint_progress()
        if self.all_goals_reached:
            rospy.loginfo("已到达所有路点，停止导航")
            self.publish_cmd_vel(0.0, 0.0)
            self.navigation_active = False
            return
            
        target_x, target_y = self.get_target_point()
        if target_x is None or target_y is None:
            rospy.logwarn_throttle(5.0, "无法确定局部目标点")
            return
        
        robot_state = (0.0, 0.0, 0.0)
        goal_state = (target_x, target_y)
        
        current_goal_num = self.current_waypoint_index + 1
        total_goals = len(self.waypoints)
        rospy.loginfo(
            f"导航至目标点 {current_goal_num}/{total_goals}: 当前位置 ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
            f"当前路点(世界): ({self.goal_x:.2f}, {self.goal_y:.2f}), "
            f"局部前瞻目标(局部): ({target_x:.2f}, {target_y:.2f})")

        self.update_mppi_maps()

        try:
            start_time = time.time()
            control_sequence = self.mppi.solve(robot_state, goal_state)
            end_time = time.time()
            
            # 记录计算时间
            compute_time = end_time - start_time
            controller_name = "MPPI+NMPC" if self.use_nmpc else "MPPI"
            rospy.logdebug(f"{controller_name}计算时间: {compute_time:.4f}秒")

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
            rospy.logerr(f"MPPI+NMPC计算错误: {e}")
            self.publish_cmd_vel(0.0, 0.0)

        # dx = target_x   # 局部坐标下的目标点
        # dy = target_y
        # angle_to_goal = math.atan2(dy, dx)              # 需要转的角度
        # dist_to_goal = math.hypot(dx, dy)               # 和目标的距离

        # # 线速度：距离越远越快，限制在 [v_min, v_max]
        # v = 0.5 * dist_to_goal
        # v = max(self.v_min, min(self.v_max, v))
        # v = v * max(0.1, 1.0 - abs(angle_to_goal) / math.pi)

        # # 角速度：按角度比例转向，限制在 [w_min, w_max]
        # w = 0.5 * angle_to_goal
        # w = max(self.w_min, min(self.w_max, w))

        # rospy.loginfo(f"[P_CTRL] dx={dx:.2f}, dy={dy:.2f}, dist={dist_to_goal:.2f}, "
        #               f"angle={angle_to_goal:.2f}, v={v:.2f}, w={w:.2f}")

        # if self.goal_reached or self.stopped:
        #     self.publish_cmd_vel(0.0, 0.0)
        #     return

        # self.publish_cmd_vel(v, w)

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
        """发布速度控制命令，并缓存最近控制用于延迟补偿"""
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_vel_pub.publish(cmd)

        # 缓存最近一次控制
        self.last_cmd_v = v
        self.last_cmd_w = w
        self.last_cmd_time = rospy.Time.now()
         
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