# lyx_trav_safety_orin

面向野外移动机器人导航的地形通过性与语义安全预测。

本仓库是一个 ROS1/catkin 工作空间，使用的是真实野外移动机器人平台 Scout 2.0，集成了 ZED2i 感知、Fixposition RTK/里程计、SCOUT 底盘驱动、局部/全局地图构建、地形通过性与语义安全预测，以及基于预测结果的实车导航流程。

## 项目结构

```text
lyx_trav_safety_orin/
├── src/
│   ├── trav_safety/        # 地形通过性与语义安全预测节点、模型结构、消息定义
│   ├── nav_realcar/        # 实车导航节点与启动文件
│   ├── map_realcar/        # 实车/离线地图构建节点与启动文件
│   ├── scout_ros/          # SCOUT 底盘 ROS 驱动相关包
│   ├── ugv_sdk/            # UGV 底盘通信 SDK
│   ├── fix_positon_ros1/   # Fixposition ROS1 驱动
│   └── zed2i/              # ZED2i ROS wrapper/examples
└── 编译指令.txt             # 原始编译与运行记录
```

## Checkpoints

模型权重文件体积较大，未纳入 Git 仓库。请从百度网盘下载后放回以下目录：

```text
src/trav_safety/src/checkpoints/
```

下载信息：

```text
通过网盘分享的文件：checkpoints
链接: https://pan.baidu.com/s/1Efd54YTGqlQpIUHTRsWZyw
提取码: 9wd9
```

当前代码中涉及的权重文件包括：

```text
double_WTConv.ckpt
safety_bev_fs.ckpt
safety_bev.ckpt
safety_final2.ckpt
STANet.ckpt
wayfaster.ckpt
```

## 环境依赖

建议环境：

- Ubuntu + ROS1 + catkin_tools
- Python 3
- Conda 环境：`lyx`
- ZED SDK 与 ZED ROS wrapper
- Fixposition ROS1 driver
- SCOUT/UGV 底盘 CAN 通信环境
- Python 依赖：`numpy`、`torch`、`opencv-python`、`cv_bridge` 等

ROS 包依赖主要包括：

- `roscpp`
- `rospy`
- `std_msgs`
- `sensor_msgs`
- `geometry_msgs`
- `nav_msgs`
- `tf`
- `tf2`
- `tf2_ros`
- `cv_bridge`
- `image_transport`
- `message_filters`
- `message_generation`
- `message_runtime`

## 编译

在工作空间根目录执行：

```bash
catkin build scout_base scout_bringup scout_description scout_msgs ugv_sdk trav_safety nav_realcar map_realcar fixposition_driver_ros1 zed_wrapper -DPYTHON_EXECUTABLE=/usr/bin/python3
```

如果底层驱动已编译，也可以只编译核心功能包：

```bash
catkin build trav_safety nav_realcar map_realcar fixposition_driver_ros1 zed_wrapper -DPYTHON_EXECUTABLE=/usr/bin/python3
```

编译完成后：

```bash
source devel/setup.bash
```

## 地图构建

先进入工作空间并加载环境：

```bash
cd lyx_trav_safety_orin
conda activate lyx
source devel/setup.bash
```

终端 1 启动地图节点：

```bash
./src/map_realcar/launch/start_map_realcar.sh
```

局部地图感知稳定后，在终端 2 控制全局地图采集：

```bash
rosservice call /start_cap
rosservice call /end_cap
```

## 实车导航

实验前建议先将小车开到开阔区域，通过浏览器访问 RTK 设备 IP 检查 Fixposition 是否正常运行。第一次有效解位置会作为原点。随后依次开往实验路点，通过以下命令记录 ENU 坐标，并写入 `start_realcar.launch`：

```bash
rostopic echo /fixposition/odometry_enu
```

启动前检查相机与 RTK：

```bash
cd lyx_trav_safety_orin
conda activate lyx
source devel/setup.bash
roslaunch nav_realcar test_before_start.launch
```

检查底盘 CAN 通信：

```bash
sudo modprobe gs_usb
sudo ip link set can0 up type can bitrate 500000
roslaunch scout_base scout_base.launch
roslaunch scout_bringup scout_teleop_keyboard.launch
```

正式导航：

```bash
./src/nav_realcar/launch/start_realcar.sh
rosservice call /start_navigation
```

## 安全头/非安全头模型切换

切换模型时需要同步修改以下位置：

1. 修改 `src/nav_realcar/launch/start_realcar.launch` 中的 `safety_head` 参数，并切换为对应 checkpoint。
2. 修改 `src/trav_safety/scripts/trav_safety_node.py` 中的推理模块 import。
3. 修改 `src/trav_safety/src/infer/` 中对应模块里的 `TravNet` 导入。

## 说明

- `src/trav_safety/src/checkpoints/` 已被 `.gitignore` 排除，请通过网盘单独下载。
- `build/`、`devel/`、`logs/`、`.catkin_tools/`、`__pycache__/` 等本地构建与缓存文件不会提交。
- 本仓库中的 `scout_ros`、`ugv_sdk`、`zed2i`、`fix_positon_ros1` 包含第三方驱动或 SDK 代码，请根据各自上游项目的许可协议使用。
