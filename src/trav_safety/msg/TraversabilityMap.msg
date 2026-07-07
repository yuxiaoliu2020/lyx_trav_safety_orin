# 通过性地图消息定义
Header header            # 标准头信息，包含时间戳和坐标帧
float32[] mu_data        # 直线通过性数据（展平的2D栅格）
float32[] nu_data        # 角度通过性数据（展平的2D栅格）
uint32 width             # 栅格宽度
uint32 height            # 栅格高度
float32 resolution       # 地图分辨率（米/单元格）
geometry_msgs/Pose origin # 地图原点相对于frame_id的位置