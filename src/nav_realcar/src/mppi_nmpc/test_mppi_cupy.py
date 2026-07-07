import numpy as np
import os
import matplotlib.pyplot as plt
import cupy as cp  # 导入cupy
from nav_realcar.src.mppi_nmpc.lyx_mppi_cupy_0114 import MPPI_CuPy, Config

if __name__ == '__main__':
    # 1. 从文件加载牵引力地图
    trav_map_path = os.path.join(os.path.dirname(__file__), 'trav_map_0.npy')
    traction_map_orig = np.load(trav_map_path)
    print(f"成功加载牵引力地图，形状为: {traction_map_orig.shape}")

    # 确保地图是float32类型
    if traction_map_orig.dtype != np.float32:
        traction_map_orig = traction_map_orig.astype(np.float32)
        
    # 2. 从文件加载风险地图
    risk_map_path = os.path.join(os.path.dirname(__file__), 'risk_map.npy')
    try:
        risk_map = np.load(risk_map_path)
        print(f"成功加载风险地图，形状为: {risk_map.shape}")
        
        # 确保地图是float32类型
        if risk_map.dtype != np.float32:
            risk_map = risk_map.astype(np.float32)
        
    except Exception as e:
        print(f"加载风险地图出错: {e}")
        print("使用默认风险地图...")
        # 创建默认风险地图（全0表示无风险）
        risk_map = np.zeros((traction_map_orig.shape[2], traction_map_orig.shape[3]), dtype=np.float32)

    # 获取地图尺寸
    map_height, map_width = traction_map_orig.shape[2:]
    
    # 3. 重新格式化牵引力地图为CuPy版本需要的格式 (H, W, 2)
    traction_map = np.zeros((map_height, map_width, 2), dtype=np.float32)
    traction_map[:, :, 0] = traction_map_orig[0, 0]  # 线性牵引力
    traction_map[:, :, 1] = traction_map_orig[0, 1]  # 角度牵引力
    
    # 确保风险地图维度匹配 (H, W)
    if len(risk_map.shape) == 3:
        risk_map = risk_map[0]  # 取第一个通道
    
    # 4. 创建配置对象 - 使用CuPy版本的Config类
    cfg = Config(
        horizon=100,          # 对应原来的num_steps
        dt=0.1,               # 时间步长
        num_samples=1024,     # 对应原来的num_control_rollouts
        num_iterations=5,     # 优化迭代次数
        
        noise_sigma_v=0.5,    # 线速度噪声标准差
        noise_sigma_w=1.0,    # 角速度噪声标准差
        
        v_min=0.0,            # 线速度最小值
        v_max=3.0,            # 线速度最大值
        w_min=-np.pi,         # 角速度最小值
        w_max=np.pi,          # 角速度最大值
        
        lambda_=1.0,          # 温度参数，对应原来的lambda_weight
        dist_weight=1.0,      # 距离代价权重
        risk_cost=5e2,        # 风险代价权重
        
        map_resolution=1.0,   # 地图分辨率
        seed=42               # 随机数种子
    )

    # 5. 初始化MPPI规划器 - 直接传入地图和配置
    mppi_planner = MPPI_CuPy(traction_map, risk_map, cfg)
    
    # 6. 设置起点和目标
    start_x, start_y = map_width*0.5, map_height*0.2  
    goal_x, goal_y = map_width*0.5, map_height*0.9

    # 初始状态和目标
    x0 = (start_x, start_y, np.pi/4)  # 元组形式，适应CuPy版本API
    xgoal = (goal_x, goal_y)          # 元组形式
    
    # 目标容忍度（用于判断是否到达目标）
    goal_tolerance = 2.0

    # 7. 迭代模拟和可视化
    max_steps = 1000  # 最大模拟步数
    xhist = np.zeros((max_steps+1, 3))*np.nan  # 状态历史
    uhist = np.zeros((max_steps, 2))*np.nan    # 控制输入历史
    xhist[0] = np.array(x0)  # 设置初始状态

    plot_every_n = 200  # 每200步可视化一次
    
    print("开始迭代模拟...")
    for t in range(max_steps):
        # 求解MPPI获取控制序列（已转换为numpy数组）
        useq = mppi_planner.solve(x0, xgoal)
        u_curr = useq[0]  # 取第一个控制输入
        uhist[t] = u_curr  # 记录控制输入历史

        if t == 0:
            print(f"初始状态: x0 = {x0}")
            print(f"目标位置: xgoal = {xgoal}")
            print(f"控制序列第一步: u_curr = {u_curr}")
        
        # 模拟机器人状态前向运动
        # 从当前位置获取牵引力
            # 从当前位置获取牵引力时增加安全检查
        x_pos = float(xhist[t, 0])
        y_pos = float(xhist[t, 1])
        
        if np.isnan(x_pos) or np.isnan(y_pos):
            print(f"警告: 步骤 {t} 的位置包含 NaN!")
        
        xi = int(np.clip(x_pos, 0, map_width-1))
        yi = int(np.clip(y_pos, 0, map_height-1))
        
        # xi = int(xhist[t, 0])
        # yi = int(xhist[t, 1])
        # xi = max(0, min(map_width-1, xi))
        # yi = max(0, min(map_height-1, yi))
        
        lin_traction = traction_map[yi, xi, 0]  # 注意索引顺序变为[y, x, 0]
        ang_traction = traction_map[yi, xi, 1]  # 注意索引顺序变为[y, x, 1]
        
        # 应用动力学模型更新状态
        xhist[t+1, 0] = xhist[t, 0] + cfg.dt * lin_traction * u_curr[0] * np.cos(xhist[t, 2])
        xhist[t+1, 1] = xhist[t, 1] + cfg.dt * lin_traction * u_curr[0] * np.sin(xhist[t, 2])
        xhist[t+1, 2] = xhist[t, 2] + cfg.dt * ang_traction * u_curr[1]
        
        # 更新初始状态用于下一次规划
        x0 = (float(xhist[t+1, 0]), float(xhist[t+1, 1]), float(xhist[t+1, 2]))
        
        # 周期性可视化
        if t % plot_every_n == 0:
            print(f"步骤 {t}: 当前位置 = ({xhist[t+1, 0]:.2f}, {xhist[t+1, 1]:.2f})")
            
            # 创建可视化图形
            plt.figure(figsize=(12, 10))
            
            # 子图1: 牵引力地图(线性)和轨迹
            plt.subplot(2, 2, 1)
            plt.imshow(traction_map[:, :, 0], origin='lower', cmap='viridis')
            plt.colorbar(label='Line Traction')
            
            # 绘制起点和当前位置
            plt.plot(xhist[0, 0], xhist[0, 1], 'ro', markersize=10, markerfacecolor='none', label="Start")
            plt.plot(xhist[t+1, 0], xhist[t+1, 1], 'ro', markersize=10, label="Curr Location", zorder=5)
            
            # 绘制目标区域
            goal_circle = plt.Circle((xgoal[0], xgoal[1]), goal_tolerance, 
                                    color='b', fill=False, label="Goal", zorder=6)
            plt.gca().add_patch(goal_circle)
            
            # 绘制历史轨迹
            valid_hist = ~np.isnan(xhist[:t+2, 0])
            plt.plot(xhist[:t+2, 0][valid_hist], xhist[:t+2, 1][valid_hist], 'r-', label="History Traj")
            
            plt.title('Line Traction Map and History Trajectory')
            plt.legend()
            
            # 子图2: 角度牵引力地图和轨迹
            plt.subplot(2, 2, 2)
            plt.imshow(traction_map[:, :, 1], origin='lower', cmap='viridis')
            plt.colorbar(label='Angular Traction')

            # 绘制起点和当前位置
            plt.plot(xhist[0, 0], xhist[0, 1], 'ro', markersize=10, markerfacecolor='none', label="Start")
            plt.plot(xhist[t+1, 0], xhist[t+1, 1], 'ro', markersize=10, label="Curr Location", zorder=5)

            # 绘制目标区域
            goal_circle = plt.Circle((xgoal[0], xgoal[1]), goal_tolerance, 
                                    color='b', fill=False, label="Goal", zorder=6)
            plt.gca().add_patch(goal_circle)

            # 绘制历史轨迹
            plt.plot(xhist[:t+2, 0][valid_hist], xhist[:t+2, 1][valid_hist], 'r-', label="History Traj")

            plt.title('Angular Traction Map and History Trajectory')
            plt.legend()

            # 子图3: 风险地图与轨迹
            plt.subplot(2, 2, 3)
            plt.imshow(risk_map, origin='lower', cmap='plasma')
            plt.colorbar(label='Risk')

            # 绘制起点和当前位置
            plt.plot(xhist[0, 0], xhist[0, 1], 'ro', markersize=10, markerfacecolor='none', label="Start")
            plt.plot(xhist[t+1, 0], xhist[t+1, 1], 'ro', markersize=10, label="Curr Location", zorder=5)

            # 绘制目标区域
            goal_circle = plt.Circle((xgoal[0], xgoal[1]), goal_tolerance, 
                                    color='b', fill=False, label="Goal", zorder=6)
            plt.gca().add_patch(goal_circle)

            # 绘制历史轨迹
            plt.plot(xhist[:t+2, 0][valid_hist], xhist[:t+2, 1][valid_hist], 'r-', label="History Traj")

            plt.title('Risk Map and History Trajectory')
            plt.legend()
            
            # 子图4: 控制输入历史
            plt.subplot(2, 2, 4)
            valid_hist_u = ~np.isnan(uhist[:t+1, 0])
            plt.plot(np.arange(t+1)[valid_hist_u], uhist[:t+1, 0][valid_hist_u], 'b-', label='Linear(v)')
            plt.plot(np.arange(t+1)[valid_hist_u], uhist[:t+1, 1][valid_hist_u], 'r-', label='Angular(w)')
            plt.grid(True)
            plt.xlabel('Step')
            plt.ylabel('Control Input')
            plt.title('Historical Control Inputs')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'cupy_mppi_step_{t}.png')
            plt.close()
            print(f"步骤 {t} 的可视化已保存为 'cupy_mppi_step_{t}.png'")
        
        # 移动规划窗口（滚动时域）
        mppi_planner.shift_and_update(1)
        
        # 检查是否到达目标
        dist_to_goal = np.linalg.norm(xhist[t+1, :2] - np.array(xgoal))
        if dist_to_goal <= goal_tolerance:
            print(f"目标在 t={t*cfg.dt:.2f}秒时到达!")
            
            # 在到达目标时进行额外可视化
            print(f"步骤 {t}: 当前位置 = ({xhist[t+1, 0]:.2f}, {xhist[t+1, 1]:.2f}), 已到达目标!")
            
            # 创建可视化图形
            plt.figure(figsize=(12, 10))
            
            # 子图1: 牵引力地图(线性)和轨迹
            plt.subplot(2, 2, 1)
            plt.imshow(traction_map[:, :, 0], origin='lower', cmap='viridis')
            plt.colorbar(label='Line Traction')
            
            # 绘制起点和当前位置
            plt.plot(xhist[0, 0], xhist[0, 1], 'ro', markersize=10, markerfacecolor='none', label="Start")
            plt.plot(xhist[t+1, 0], xhist[t+1, 1], 'ro', markersize=10, label="Goal Reached!", zorder=5)
            
            # 绘制目标区域
            goal_circle = plt.Circle((xgoal[0], xgoal[1]), goal_tolerance, 
                                    color='b', fill=False, label="Goal", zorder=6)
            plt.gca().add_patch(goal_circle)
            
            # 绘制历史轨迹
            valid_hist = ~np.isnan(xhist[:t+2, 0])
            plt.plot(xhist[:t+2, 0][valid_hist], xhist[:t+2, 1][valid_hist], 'r-', label="History Traj")
            
            plt.title('Line Traction Map and History Trajectory')
            plt.legend()
            
            # 子图2: 角度牵引力地图和轨迹
            plt.subplot(2, 2, 2)
            plt.imshow(traction_map[:, :, 1], origin='lower', cmap='viridis')
            plt.colorbar(label='Angular Traction')

            plt.plot(xhist[0, 0], xhist[0, 1], 'ro', markersize=10, markerfacecolor='none', label="Start")
            plt.plot(xhist[t+1, 0], xhist[t+1, 1], 'ro', markersize=10, label="Goal Reached!", zorder=5)
            goal_circle = plt.Circle((xgoal[0], xgoal[1]), goal_tolerance, 
                                    color='b', fill=False, label="Goal", zorder=6)
            plt.gca().add_patch(goal_circle)
            plt.plot(xhist[:t+2, 0][valid_hist], xhist[:t+2, 1][valid_hist], 'r-', label="History Traj")
            plt.title('Angular Traction Map and History Trajectory')
            plt.legend()

            # 子图3: 风险地图与轨迹
            plt.subplot(2, 2, 3)
            plt.imshow(risk_map, origin='lower', cmap='plasma')
            plt.colorbar(label='Risk')

            plt.plot(xhist[0, 0], xhist[0, 1], 'ro', markersize=10, markerfacecolor='none', label="Start")
            plt.plot(xhist[t+1, 0], xhist[t+1, 1], 'ro', markersize=10, label="Goal Reached!", zorder=5)
            goal_circle = plt.Circle((xgoal[0], xgoal[1]), goal_tolerance, 
                                    color='b', fill=False, label="Goal", zorder=6)
            plt.gca().add_patch(goal_circle)
            plt.plot(xhist[:t+2, 0][valid_hist], xhist[:t+2, 1][valid_hist], 'r-', label="History Traj")
            plt.title('Risk Map and History Trajectory')
            plt.legend()
            
            # 子图4: 控制输入历史
            plt.subplot(2, 2, 4)
            valid_hist_u = ~np.isnan(uhist[:t+1, 0])
            plt.plot(np.arange(t+1)[valid_hist_u], uhist[:t+1, 0][valid_hist_u], 'b-', label='Linear(v)')
            plt.plot(np.arange(t+1)[valid_hist_u], uhist[:t+1, 1][valid_hist_u], 'r-', label='Angular(w)')
            plt.grid(True)
            plt.xlabel('Step')
            plt.ylabel('Control Input')
            plt.title('Historical Control Inputs')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'cupy_mppi_goal_reached_step_{t}.png')
            plt.close()
            print(f"目标到达时的可视化已保存为 'cupy_mppi_goal_reached_step_{t}.png'")
            
            break
    
    # 绘制完整控制输入历史
    plt.figure(figsize=(12, 4))
    valid_hist_u = ~np.isnan(uhist[:, 0])
    plt.plot(np.arange(max_steps)[valid_hist_u], uhist[:, 0][valid_hist_u], 'b-', label='Linear(v)')
    plt.plot(np.arange(max_steps)[valid_hist_u], uhist[:, 1][valid_hist_u], 'r-', label='Angular(w)')
    plt.grid(True)
    plt.xlabel('Step')
    plt.ylabel('Control Input')
    plt.title('CuPy MPPI Control History')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cupy_mppi_control_history.png')
    print("完整控制输入历史已保存为 'cupy_mppi_control_history.png'")