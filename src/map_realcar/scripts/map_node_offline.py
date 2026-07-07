#! /usr/bin/env python
import rospy
import numpy as np
import math
import tf
import os
from collections import deque
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from trav_safety.msg import SafetyMap, TraversabilityMap
from std_srvs.srv import Empty, EmptyResponse


class GlobalMappingOfflineNode:
	"""离线建图节点：在 rosbag 回放时按运动量触发融合。

	触发条件：小车自上次融合后行进距离≥fuse_dist_thresh 或 航向变化≥fuse_yaw_thresh。
	其他逻辑尽量与在线版保持一致。
	"""

	def __init__(self):
		rospy.init_node("global_mapping_offline_node")

		# 参数
		self.local_map_topic = rospy.get_param("~safety_map_topic", "/safety_map")
		self.trav_map_topic = rospy.get_param("~traversability_map_topic", "/traversability_map")
		self.pose_topic = rospy.get_param("~pose_topic", "/fixposition/odometry_enu")
		self.global_frame = rospy.get_param("~global_frame", "map")
		self.global_resolution = rospy.get_param("~global_resolution", 0.1)
		self.global_size = rospy.get_param("~global_size", 6000)
		span = self.global_size * self.global_resolution
		default_origin = -0.5 * span
		self.global_origin_x = rospy.get_param("~global_origin_x", default_origin)
		self.global_origin_y = rospy.get_param("~global_origin_y", default_origin)

		# 运动触发阈值
		self.fuse_dist_thresh = rospy.get_param("~fuse_dist_thresh", 2.0)  # 米
		yaw_deg = rospy.get_param("~fuse_yaw_thresh_deg", 15.0)
		self.fuse_yaw_thresh = math.radians(yaw_deg)

		self.roi_front_min = rospy.get_param("~roi_front_min", 0.0)
		self.roi_front_max = rospy.get_param("~roi_front_max", 8.0)

		self.publish_rate = rospy.get_param("~publish_rate", 1.0)
		self.traj_min_dist = rospy.get_param("~traj_min_dist", 0.05)

		# 保存输出配置
		self.save_outputs = rospy.get_param("~save_outputs", True)
		self.save_dir = rospy.get_param("~save_dir", os.getcwd())
		self.save_interval = rospy.get_param("~save_interval", 10.0)
		self._last_save_time = None

		# 采集开关：通过服务 start_cap / end_cap 控制
		self.capture_enabled = False

		# 可选：如果上游 origin 填错，可用此参数覆盖 [x, y]
		self.local_origin_override = rospy.get_param("~local_origin_override", [-2.0, -5.0])

		# 缓存
		self.odom_buffer = deque(maxlen=400)
		self.current_pose = None
		self.current_yaw = 0.0
		self.latest_safety_msg = None
		self.latest_trav_msg = None
		self.latest_safety_arr = None
		self.latest_trav_mu = None
		self.latest_trav_nu = None

		# 轨迹记录（世界系）
		self.world_traj = []
		self.world_traj_vx = []
		self.world_traj_vy = []
		self.world_traj_wz = []
		self.world_traj_stamp = []
		self.last_traj_x = None
		self.last_traj_y = None

		# 全局地图存储为“加权和”，配合 count 做均值；count=0 视为 unknown
		self.global_map_safety = np.zeros((self.global_size, self.global_size), dtype=np.float32)
		self.global_map_mu = np.zeros((self.global_size, self.global_size), dtype=np.float32)
		self.global_map_nu = np.zeros((self.global_size, self.global_size), dtype=np.float32)
		self.global_count_safety = np.zeros((self.global_size, self.global_size), dtype=np.float32)
		self.global_count_mu = np.zeros((self.global_size, self.global_size), dtype=np.float32)
		self.global_count_nu = np.zeros((self.global_size, self.global_size), dtype=np.float32)

		# 上一次融合时的姿态
		self.last_fuse_x = None
		self.last_fuse_y = None
		self.last_fuse_yaw = None

		# 订阅与发布
		rospy.Subscriber(self.pose_topic, Odometry, self.odom_callback, queue_size=50)
		rospy.Subscriber(self.local_map_topic, SafetyMap, self.map_callback, queue_size=10)
		rospy.Subscriber(self.trav_map_topic, TraversabilityMap, self.trav_map_callback, queue_size=10)

		self.global_map_pub = rospy.Publisher("/global_safety_map", OccupancyGrid, queue_size=1)
		self.global_mu_pub = rospy.Publisher("/global_trav_mu_map", OccupancyGrid, queue_size=1)
		self.global_nu_pub = rospy.Publisher("/global_trav_nu_map", OccupancyGrid, queue_size=1)
		self.local_map_pub = rospy.Publisher("/local_trav_mu_map", OccupancyGrid, queue_size=1)
		self.pose_vis_pub = rospy.Publisher("/global_mapper_pose", PoseStamped, queue_size=1)
		self.traj_pub = rospy.Publisher("/global_mapper_traj", Path, queue_size=1)

		# 服务：控制采集开始/结束
		rospy.Service("/start_cap", Empty, self.start_capture_srv)
		rospy.Service("/end_cap", Empty, self.end_capture_srv)

		# 定时器
		control_rate = rospy.get_param("~control_rate", 10.0)
		rospy.Timer(rospy.Duration(1.0 / control_rate), self.control_loop)
		rospy.Timer(rospy.Duration(1.0 / max(0.1, self.publish_rate)), self.publish_global_map)

		rospy.loginfo("Global mapping offline node ready: waiting for odom and local safety maps...")

	# ----------------- 基础工具 -----------------
	def _pose_yaw_from_pose(self, pose):
		q = pose.orientation
		_, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
		return yaw

	def _find_nearest_pose(self, stamp):
		if not self.odom_buffer:
			return None, None
		best = min(self.odom_buffer, key=lambda it: abs((it[0] - stamp).to_sec()))
		return best[1], best[2]

	def _append_traj(self, msg):
		if not self.capture_enabled:
			return
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		vx = msg.twist.twist.linear.x
		vy = msg.twist.twist.linear.y
		wz = msg.twist.twist.angular.z

		need_append = False
		if self.last_traj_x is None:
			need_append = True
		else:
			dx = x - self.last_traj_x
			dy = y - self.last_traj_y
			if math.hypot(dx, dy) >= self.traj_min_dist:
				need_append = True

		if not need_append:
			return

		self.last_traj_x = x
		self.last_traj_y = y

		pose_stamped = PoseStamped()
		pose_stamped.header.stamp = msg.header.stamp
		pose_stamped.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.global_frame
		pose_stamped.pose = msg.pose.pose

		self.world_traj.append(pose_stamped)
		self.world_traj_vx.append(vx)
		self.world_traj_vy.append(vy)
		self.world_traj_wz.append(wz)
		self.world_traj_stamp.append(msg.header.stamp.to_sec())

		path_msg = Path()
		path_msg.header.stamp = msg.header.stamp
		path_msg.header.frame_id = pose_stamped.header.frame_id
		path_msg.poses = self.world_traj

		self.traj_pub.publish(path_msg)

	def _publish_pose_vis(self, msg):
		pose_vis = PoseStamped()
		pose_vis.header.stamp = msg.header.stamp
		pose_vis.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.global_frame
		pose_vis.pose = msg.pose.pose
		self.pose_vis_pub.publish(pose_vis)

	# ----------------- 回调 -----------------
	def odom_callback(self, msg):
		self.current_pose = msg.pose.pose
		self.current_yaw = self._pose_yaw_from_pose(msg.pose.pose)

		self.odom_buffer.append((msg.header.stamp, msg.pose.pose, self.current_yaw))

		self._append_traj(msg)
		self._publish_pose_vis(msg)

	def map_callback(self, msg):
		self.latest_safety_msg = msg
		self.latest_safety_arr = np.array(msg.safety_data, dtype=np.float32).reshape(msg.height, msg.width)

	def trav_map_callback(self, msg):
		self.latest_trav_msg = msg
		h = msg.height
		w = msg.width
		self.latest_trav_mu = np.array(msg.mu_data, dtype=np.float32).reshape(h, w)
		self.latest_trav_nu = np.array(msg.nu_data, dtype=np.float32).reshape(h, w)
		self.publish_local_trav_mu(msg)

	def publish_local_trav_mu(self, msg):
		"""将最新局部 traversability mu 图转为 OccupancyGrid，并固定在 base_link 方便 RViz 观察。"""
		grid = OccupancyGrid()
		grid.header.stamp = msg.header.stamp
		grid.header.frame_id = "base_link"

		grid.info.resolution = msg.resolution
		grid.info.width = msg.width
		grid.info.height = msg.height
		grid.info.origin.position.x = msg.origin.position.x
		grid.info.origin.position.y = msg.origin.position.y
		grid.info.origin.position.z = msg.origin.position.z
		grid.info.origin.orientation = msg.origin.orientation

		flat = np.array(msg.mu_data, dtype=np.float32)
		out = np.full_like(flat, -1, dtype=np.int8)

		# 语义：mu ∈ [0,1]，0=最不可通过，1=最易通过
		# OccupancyGrid: 0=free(白)，100=occupied(黑)
		# 为了“大白小黑”，做反转映射：mu=1 -> occ=0；mu=0 -> occ=100
		valid = flat >= 0.0
		if np.any(valid):
			mu = np.clip(flat[valid], 0.0, 1.0)
			occ = 1.0 - mu
			out[valid] = (occ * 100.0).astype(np.int8)
		grid.data = out.tolist()

		self.local_map_pub.publish(grid)

	# ----------------- 主循环 -----------------
	def control_loop(self, _event):
		if not self.capture_enabled:
			return
		if self.current_pose is None or self.latest_safety_msg is None or self.latest_trav_msg is None:
			return

		x = self.current_pose.position.x
		y = self.current_pose.position.y
		yaw = self.current_yaw

		need_fuse = False
		if self.last_fuse_x is None:
			need_fuse = True
		else:
			dist = math.hypot(x - self.last_fuse_x, y - self.last_fuse_y)
			dyaw = abs(math.atan2(math.sin(yaw - self.last_fuse_yaw), math.cos(yaw - self.last_fuse_yaw)))
			if dist >= self.fuse_dist_thresh or dyaw >= self.fuse_yaw_thresh:
				need_fuse = True

		if not need_fuse:
			return

		success = self.try_mapping()
		if success:
			self.last_fuse_x = x
			self.last_fuse_y = y
			self.last_fuse_yaw = yaw

	# ----------------- 采集与融合 -----------------
	def try_mapping(self):
		s_msg = self.latest_safety_msg
		t_msg = self.latest_trav_msg
		if (
			s_msg is None
			or t_msg is None
			or self.latest_safety_arr is None
			or self.latest_trav_mu is None
			or self.latest_trav_nu is None
		):
			rospy.logwarn("No local maps available for mapping")
			return False

		# 选择时间更近的一帧用来找姿态
		ts_ref = s_msg.header.stamp if s_msg.header.stamp > t_msg.header.stamp else t_msg.header.stamp
		pose_at_map, yaw_at_map = self._find_nearest_pose(ts_ref)
		if pose_at_map is None:
			rospy.logwarn("No pose available near map stamp; skip mapping")
			return False

		fused_s = self.fuse_local_to_global(s_msg, self.latest_safety_arr, pose_at_map, yaw_at_map, self.global_map_safety)
		fused_mu, fused_nu = self.fuse_trav_to_global(t_msg, self.latest_trav_mu, self.latest_trav_nu, pose_at_map, yaw_at_map)

		fused_total = fused_s + fused_mu + fused_nu
		if fused_total == 0:
			rospy.logwarn("Mapping skipped: ROI empty or out of bounds")
			return False

		rospy.loginfo(
			f"Fused safety {fused_s} / mu {fused_mu} / nu {fused_nu} cells at stamp {ts_ref.to_sec():.2f}"
		)
		return True

	def fuse_local_to_global(self, map_msg, local, pose, yaw_robot, global_arr):
		height = map_msg.height
		width = map_msg.width
		res = map_msg.resolution

		ox = map_msg.origin.position.x
		oy = map_msg.origin.position.y
		if len(self.local_origin_override) == 2:
			ox, oy = float(self.local_origin_override[0]), float(self.local_origin_override[1])
		oyaw = self._pose_yaw_from_pose(map_msg.origin)

		xs = (np.arange(width, dtype=np.float32) + 0.5) * res
		ys = (np.arange(height, dtype=np.float32) + 0.5) * res
		xx = xs[None, :]
		yy = ys[:, None]

		cos_o = math.cos(oyaw)
		sin_o = math.sin(oyaw)
		lx = ox + xx * cos_o - yy * sin_o
		ly = oy + xx * sin_o + yy * cos_o

		rx = pose.position.x
		ry = pose.position.y
		cos_r = math.cos(yaw_robot)
		sin_r = math.sin(yaw_robot)

		frame_id = map_msg.header.frame_id if map_msg.header.frame_id else "base_link"
		is_local_frame = frame_id in ("base_link", "base", "base_footprint")

		if is_local_frame:
			front_proj = lx
			dist = np.hypot(lx, ly)
			wx = rx + lx * cos_r - ly * sin_r
			wy = ry + lx * sin_r + ly * cos_r
		else:
			wx = lx
			wy = ly
			dx = wx - rx
			dy = wy - ry
			front_proj = dx * cos_r + dy * sin_r
			dist = np.hypot(dx, dy)

		mask = (front_proj >= self.roi_front_min) & (front_proj <= self.roi_front_max) & (dist <= self.roi_front_max)

		if not np.any(mask):
			return 0

		gx_f = (wx - self.global_origin_x) / self.global_resolution
		gy_f = (wy - self.global_origin_y) / self.global_resolution

		mask &= (local >= 0.0)
		if not np.any(mask):
			return 0

		vals = local[mask]
		gx_f = gx_f[mask]
		gy_f = gy_f[mask]

		fused, bbox = self._bilinear_accumulate(vals, gx_f, gy_f, self.global_map_safety, self.global_count_safety)
		if fused > 0:
			self._apply_closing(bbox)
		return fused

	def fuse_trav_to_global(self, map_msg, mu_arr, nu_arr, pose, yaw_robot):
		height = map_msg.height
		width = map_msg.width
		res = map_msg.resolution

		ox = map_msg.origin.position.x
		oy = map_msg.origin.position.y
		if len(self.local_origin_override) == 2:
			ox, oy = float(self.local_origin_override[0]), float(self.local_origin_override[1])
		oyaw = self._pose_yaw_from_pose(map_msg.origin)

		xs = (np.arange(width, dtype=np.float32) + 0.5) * res
		ys = (np.arange(height, dtype=np.float32) + 0.5) * res
		xx = xs[None, :]
		yy = ys[:, None]

		cos_o = math.cos(oyaw)
		sin_o = math.sin(oyaw)
		lx = ox + xx * cos_o - yy * sin_o
		ly = oy + xx * sin_o + yy * cos_o

		rx = pose.position.x
		ry = pose.position.y
		cos_r = math.cos(yaw_robot)
		sin_r = math.sin(yaw_robot)

		frame_id = map_msg.header.frame_id if map_msg.header.frame_id else "base_link"
		is_local_frame = frame_id in ("base_link", "base", "base_footprint")

		if is_local_frame:
			front_proj = lx
			dist = np.hypot(lx, ly)
			wx = rx + lx * cos_r - ly * sin_r
			wy = ry + lx * sin_r + ly * cos_r
		else:
			wx = lx
			wy = ly
			dx = wx - rx
			dy = wy - ry
			front_proj = dx * cos_r + dy * sin_r
			dist = np.hypot(dx, dy)

		mask = (front_proj >= self.roi_front_min) & (front_proj <= self.roi_front_max) & (dist <= self.roi_front_max)
		if not np.any(mask):
			return 0, 0

		gx_f = (wx - self.global_origin_x) / self.global_resolution
		gy_f = (wy - self.global_origin_y) / self.global_resolution

		mask &= (mu_arr >= 0.0) | (nu_arr >= 0.0)
		if not np.any(mask):
			return 0, 0

		mu_mask = mask & (mu_arr >= 0.0)
		nu_mask = mask & (nu_arr >= 0.0)

		fused_mu = 0
		fused_nu = 0

		bbox = None
		if np.any(mu_mask):
			fused_mu, bbox = self._bilinear_accumulate(mu_arr[mu_mask], gx_f[mu_mask], gy_f[mu_mask], self.global_map_mu, self.global_count_mu)

		if np.any(nu_mask):
			fused_nu, bbox_nu = self._bilinear_accumulate(nu_arr[nu_mask], gx_f[nu_mask], gy_f[nu_mask], self.global_map_nu, self.global_count_nu)
			if bbox is None:
				bbox = bbox_nu
			elif bbox_nu is not None:
				x0 = min(bbox[0], bbox_nu[0])
				y0 = min(bbox[1], bbox_nu[1])
				x1 = max(bbox[2], bbox_nu[2])
				y1 = max(bbox[3], bbox_nu[3])
				bbox = (x0, y0, x1, y1)

		if bbox is not None:
			self._apply_closing(bbox)

		return fused_mu, fused_nu

	def _bilinear_accumulate(self, vals, gx_f, gy_f, sum_arr, count_arr):
		size = self.global_size
		gx0 = np.floor(gx_f).astype(np.int32)
		gy0 = np.floor(gy_f).astype(np.int32)
		gx1 = gx0 + 1
		gy1 = gy0 + 1

		dx = gx_f - gx0
		dy = gy_f - gy0

		w00 = (1.0 - dx) * (1.0 - dy)
		w01 = (1.0 - dx) * dy
		w10 = dx * (1.0 - dy)
		w11 = dx * dy

		idx_list = []
		w_list = []
		v_list = []
		bounds = []

		def collect(mask, gx, gy, w):
			if not np.any(mask):
				return
			idx = (gy[mask].astype(np.int64) * size + gx[mask].astype(np.int64)).ravel()
			idx_list.append(idx)
			w_list.append(w[mask].ravel())
			v_list.append((vals[mask] * w[mask]).ravel())
			bounds.append((gx[mask].min(), gy[mask].min(), gx[mask].max(), gy[mask].max()))

		mask00 = (gx0 >= 0) & (gx0 < size) & (gy0 >= 0) & (gy0 < size) & (w00 > 0)
		mask01 = (gx0 >= 0) & (gx0 < size) & (gy1 >= 0) & (gy1 < size) & (w01 > 0)
		mask10 = (gx1 >= 0) & (gx1 < size) & (gy0 >= 0) & (gy0 < size) & (w10 > 0)
		mask11 = (gx1 >= 0) & (gx1 < size) & (gy1 >= 0) & (gy1 < size) & (w11 > 0)

		collect(mask00, gx0, gy0, w00)
		collect(mask01, gx0, gy1, w01)
		collect(mask10, gx1, gy0, w10)
		collect(mask11, gx1, gy1, w11)

		if not idx_list:
			return 0, None

		idx_all = np.concatenate(idx_list)
		w_all = np.concatenate(w_list)
		v_all = np.concatenate(v_list)

		unique_idx, inv = np.unique(idx_all, return_inverse=True)
		w_sum = np.bincount(inv, weights=w_all)
		v_sum = np.bincount(inv, weights=v_all)

		flat_sum = sum_arr.ravel()
		flat_cnt = count_arr.ravel()

		old_cnt = flat_cnt[unique_idx]
		old_mean = np.where(old_cnt > 0.0, flat_sum[unique_idx] / old_cnt, np.inf)
		new_val = np.divide(v_sum, w_sum, out=np.zeros_like(v_sum), where=w_sum > 0)
		updated = np.minimum(old_mean, new_val)

		flat_cnt[unique_idx] = 1.0
		flat_sum[unique_idx] = updated

		x_min = min(b[0] for b in bounds)
		y_min = min(b[1] for b in bounds)
		x_max = max(b[2] for b in bounds)
		y_max = max(b[3] for b in bounds)

		return unique_idx.size, (x_min, y_min, x_max, y_max)

	def _apply_closing(self, bbox):
		if bbox is None:
			return
		x0, y0, x1, y1 = bbox
		x0 = max(0, x0 - 1)
		y0 = max(0, y0 - 1)
		x1 = min(self.global_size - 1, x1 + 1)
		y1 = min(self.global_size - 1, y1 + 1)

		slices = (slice(y0, y1 + 1), slice(x0, x1 + 1))
		count = self.global_count_safety[slices] + self.global_count_mu[slices] + self.global_count_nu[slices]
		known = count > 0.0

		if not np.any(known):
			return

		# binary closing on known mask
		kn = known.astype(np.uint8)
		# dilation: any neighbor known
		sum_kn = self._neighbor_sum(kn)
		dilated = sum_kn > 0
		# erosion: all neighbors (3x3) known
		sum_dil = self._neighbor_sum(dilated.astype(np.uint8))
		closed = sum_dil == 9

		new_fill = closed & (~known)
		if not np.any(new_fill):
			return

		# neighbor-weighted fill value from already known cells
		mean_s = self._mean_map(self.global_map_safety[slices], self.global_count_safety[slices])
		mean_mu = self._mean_map(self.global_map_mu[slices], self.global_count_mu[slices])
		mean_nu = self._mean_map(self.global_map_nu[slices], self.global_count_nu[slices])

		def fill_one(sum_arr, cnt_arr, mean_arr):
			val = np.where(np.isfinite(mean_arr), mean_arr, 0.0)
			kn_local = np.isfinite(mean_arr)
			neigh_cnt = self._neighbor_sum(kn_local.astype(np.uint8))
			neigh_sum = self._neighbor_sum(val)
			mask_fill = new_fill & (neigh_cnt >= 1)
			if not np.any(mask_fill):
				return
			fill_val = neigh_sum[mask_fill] / neigh_cnt[mask_fill]
			cnt_arr_slice = cnt_arr[slices]
			sum_arr_slice = sum_arr[slices]
			cnt_arr_slice[mask_fill] = neigh_cnt[mask_fill].astype(np.float32)
			sum_arr_slice[mask_fill] = fill_val.astype(np.float32) * cnt_arr_slice[mask_fill]

		fill_one(self.global_map_safety, self.global_count_safety, mean_s)
		fill_one(self.global_map_mu, self.global_count_mu, mean_mu)
		fill_one(self.global_map_nu, self.global_count_nu, mean_nu)

	def _neighbor_sum(self, arr):
		"""3x3 邻域求和，边界不填充。"""
		out = np.zeros_like(arr, dtype=np.float32)
		for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
			if dx > 0:
				s_row = slice(dx, None)
				d_row = slice(0, -dx)
			elif dx < 0:
				s_row = slice(0, dx)
				d_row = slice(-dx, None)
			else:
				s_row = slice(None)
				d_row = slice(None)

			if dy > 0:
				s_col = slice(dy, None)
				d_col = slice(0, -dy)
			elif dy < 0:
				s_col = slice(0, dy)
				d_col = slice(-dy, None)
			else:
				s_col = slice(None)
				d_col = slice(None)

			out[d_row, d_col] += arr[s_row, s_col].astype(np.float32)
		return out

	def _fill_unknown_holes(self, arr, min_neighbors=6):
		"""单次填充：unknown 且邻域已知数≥min_neighbors，用邻域均值填充。"""
		known = arr >= 0.0
		if not np.any(~known):
			return arr

		val = np.where(known, arr, 0.0)
		cnt = np.zeros_like(arr, dtype=np.int16)
		ssum = np.zeros_like(arr, dtype=np.float32)

		for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
			if dx > 0:
				s_row = slice(dx, None)
				d_row = slice(0, -dx)
			elif dx < 0:
				s_row = slice(0, dx)
				d_row = slice(-dx, None)
			else:
				s_row = slice(None)
				d_row = slice(None)

			if dy > 0:
				s_col = slice(dy, None)
				d_col = slice(0, -dy)
			elif dy < 0:
				s_col = slice(0, dy)
				d_col = slice(-dy, None)
			else:
				s_col = slice(None)
				d_col = slice(None)

			cnt[d_row, d_col] += known[s_row, s_col]
			ssum[d_row, d_col] += val[s_row, s_col]

		target = (~known) & (cnt >= min_neighbors)
		if not np.any(target):
			return arr

		filled = np.array(arr, copy=True)
		filled[target] = ssum[target] / cnt[target]
		return filled

	def _mean_map(self, sum_arr, count_arr):
		return np.where(count_arr > 0.0, sum_arr / count_arr, -1.0)

	# ----------------- 发布 -----------------
	def publish_global_map(self, _event):
		if not self.capture_enabled:
			return
		stamp = rospy.Time.now()
		self._publish_global_grid(self.global_map_pub, self.global_map_safety, self.global_count_safety, stamp)
		self._publish_global_grid(self.global_mu_pub, self.global_map_mu, self.global_count_mu, stamp)
		self._publish_global_grid(self.global_nu_pub, self.global_map_nu, self.global_count_nu, stamp)

		if self.save_outputs:
			self._maybe_save_outputs(stamp)

	def _publish_global_grid(self, pub, sum_arr, count_arr, stamp):
		msg = OccupancyGrid()
		msg.header.stamp = stamp
		msg.header.frame_id = self.global_frame

		msg.info.resolution = self.global_resolution
		msg.info.width = self.global_size
		msg.info.height = self.global_size
		msg.info.origin.position.x = self.global_origin_x
		msg.info.origin.position.y = self.global_origin_y
		msg.info.origin.orientation.w = 1.0

		mean_arr = self._mean_map(sum_arr, count_arr)
		arr_proc = self._fill_unknown_holes(mean_arr, min_neighbors=6)
		flat = arr_proc.ravel()
		out = np.full_like(flat, -1, dtype=np.int8)

		# 语义：arr ∈ [0,1]，0=最不安全/最难通过，1=最安全/最易通过
		# RViz 的 OccupancyGrid: 0=free(白)，100=occupied(黑)
		# 因此这里反转：safe=1 -> occ=0；safe=0 -> occ=100
		known = flat >= 0.0
		if np.any(known):
			safety = np.clip(flat[known], 0.0, 1.0)
			occ = 1.0 - safety
			out[known] = (occ * 100.0).astype(np.int8)
		msg.data = out.tolist()

		pub.publish(msg)

	def _maybe_save_outputs(self, stamp):
		if self._last_save_time is not None:
			if (stamp - self._last_save_time).to_sec() < self.save_interval:
				return
		self._last_save_time = stamp

		try:
			os.makedirs(self.save_dir, exist_ok=True)

			# 记录融合后的统计，便于排查“全 0”问题
			mu_min = float(np.min(self.global_map_mu))
			mu_max = float(np.max(self.global_map_mu))
			mu_mean = float(np.mean(self.global_map_mu))
			mu_count_sum = float(np.sum(self.global_count_mu))
			rospy.loginfo(
				f"[save_debug] mu min={mu_min:.3f} max={mu_max:.3f} mean={mu_mean:.3f} count_sum={mu_count_sum:.1f}"
			)

			# 计算均值后单次填洞，存盘时 unknown 裁剪到 0.0
			safety_mean = self._fill_unknown_holes(self._mean_map(self.global_map_safety, self.global_count_safety), min_neighbors=6)
			mu_mean = self._fill_unknown_holes(self._mean_map(self.global_map_mu, self.global_count_mu), min_neighbors=6)
			nu_mean = self._fill_unknown_holes(self._mean_map(self.global_map_nu, self.global_count_nu), min_neighbors=6)

			safety_out = np.clip(np.where(safety_mean < 0.0, 0.0, safety_mean), 0.0, 1.0)
			mu_out = np.clip(np.where(mu_mean < 0.0, 0.0, mu_mean), 0.0, 1.0)
			nu_out = np.clip(np.where(nu_mean < 0.0, 0.0, nu_mean), 0.0, 1.0)

			np.savez_compressed(
				os.path.join(self.save_dir, "global_maps.npz"),
				safety=safety_out,
				mu=mu_out,
				nu=nu_out,
				resolution=self.global_resolution,
				origin=np.array([self.global_origin_x, self.global_origin_y], dtype=np.float32),
				frame=np.array([self.global_frame]),
			)

			if self.world_traj:
				traj_arr = np.array([[p.pose.position.x, p.pose.position.y] for p in self.world_traj], dtype=np.float32)
				yaw_arr = np.array([self._pose_yaw_from_pose(p.pose) for p in self.world_traj], dtype=np.float32)
				vx_arr = np.array(self.world_traj_vx, dtype=np.float32)
				vy_arr = np.array(self.world_traj_vy, dtype=np.float32)
				wz_arr = np.array(self.world_traj_wz, dtype=np.float32)
				stamp_arr = np.array(self.world_traj_stamp, dtype=np.float64)
				np.savez_compressed(
					os.path.join(self.save_dir, "global_mapper_traj.npz"),
					xy=traj_arr,
					yaw=yaw_arr,
					vx=vx_arr,
					vy=vy_arr,
					wz=wz_arr,
					stamp=stamp_arr,
					frame=np.array([self.world_traj[0].header.frame_id]),
				)
		except Exception as e:
			rospy.logwarn(f"Saving outputs failed: {e}")

	# ----------------- 服务：采集控制 -----------------
	def _reset_global_storage(self):
		self.global_map_safety.fill(0.0)
		self.global_map_mu.fill(0.0)
		self.global_map_nu.fill(0.0)
		self.global_count_safety.fill(0.0)
		self.global_count_mu.fill(0.0)
		self.global_count_nu.fill(0.0)
		self.world_traj = []
		self.world_traj_vx = []
		self.world_traj_vy = []
		self.world_traj_wz = []
		self.world_traj_stamp = []
		self.last_traj_x = None
		self.last_traj_y = None
		self._last_save_time = None
		self.last_fuse_x = None
		self.last_fuse_y = None
		self.last_fuse_yaw = None

	def start_capture_srv(self, _req):
		self.capture_enabled = True
		self._reset_global_storage()
		rospy.loginfo("[CAPTURE] start_cap: 全局地图离线建图开始")
		return EmptyResponse()

	def end_capture_srv(self, _req):
		self.capture_enabled = False
		rospy.loginfo("[CAPTURE] end_cap: 全局地图离线建图结束")
		return EmptyResponse()

	def run(self):
		rospy.spin()


if __name__ == "__main__":
	node = GlobalMappingOfflineNode()
	node.run()
