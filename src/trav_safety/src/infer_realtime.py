import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from infer.dataloader_realtime import InputFolderDataset
from infer.infer_configs import get_cfg
from infer.infer_module import InferModule

def parse_config():
    """读取并加载配置文件"""
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify path config file')
    parser.add_argument('--input_folder', type=str, default=None, help='specify input folder path')
    args = parser.parse_args()
    
    # 加载默认配置
    config = get_cfg(args.cfg_file)
    return config, args

def visualize_results(traversability_map, pred_depth, debug, original_img, depth_img, save_path=None):
    """
    可视化推理结果，包括通过性图、原图。
    """
    print(f"Traversability map shape: {traversability_map.shape}")
    print(f"Original image shape: {original_img.shape}")
    # 通过性图显示
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
    
    trav_img_mu = traversability_map[0, 0].cpu().numpy()
    print(f"Mu值范围: min={trav_img_mu.min():.6f}, max={trav_img_mu.max():.6f}")
    ax1[0].imshow(trav_img_mu, cmap='viridis')
    ax1[0].set_title("Traversability Map (mu)")
    ax1[0].axis('off')
    
    trav_img_nu = traversability_map[0, 1].cpu().numpy()
    print(f"Nu值范围: min={trav_img_nu.min():.6f}, max={trav_img_nu.max():.6f}")
    ax1[1].imshow(trav_img_nu, cmap='viridis')
    ax1[1].set_title("Traversability Map (nu)")
    ax1[1].axis('off')
    
    plt.tight_layout()
    
    # 显示6帧RGB图像
    fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    # 取第一个batch的6帧图像
    rgb_frames = original_img[0]  # shape: (6, 3, 180, 320)
    
    for i in range(6):
        # 将tensor转换为numpy并调整通道顺序
        img = rgb_frames[i].cpu().numpy()  # (3, 180, 320)
        img = np.transpose(img, (1, 2, 0))  # (180, 320, 3)
        
        # 归一化到0-1范围
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img)
        axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig1.savefig(f'{save_path}_trav.png')
        fig2.savefig(f'{save_path}_frames.png')

    plt.show()

def main():
    # 配置读取
    configs, args = parse_config()
    print('configs:\n', configs)
    
    # 设置随机种子
    torch.manual_seed(configs.SEED)
    torch.cuda.manual_seed_all(configs.SEED)
    np.random.seed(configs.SEED)
    
    # 创建输出目录
    output_dir = 'outputs'
    travmap_dir = os.path.join(output_dir, 'travmaps')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(travmap_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    input_folder = args.input_folder if args.input_folder else 'inputs'
    print(f'使用输入文件夹 {input_folder} 进行推理...')
    infer_dataset = InputFolderDataset(configs, input_folder)
    
    # 创建推理模型
    print('Loading model...')
    model = InferModule(configs)
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)
    
    # 加载预训练模型权重
    if configs.MODEL.LOAD_NETWORK is not None:
        print('Loading saved network from {}'.format(configs.MODEL.LOAD_NETWORK))
        pretrained_dict = torch.load(configs.MODEL.LOAD_NETWORK, map_location='cpu')['state_dict']
        model.load_state_dict(pretrained_dict, strict=False)

    # 设置为评估模式
    model.eval()
    
    # 模拟实时推理 - 交互式方式
    print('Starting inference...')
    frame_idx = 0
    
    with torch.no_grad():
        # 主循环
        while True:
            print(f"\n=== Processing frame {frame_idx} ===")
            
            # 获取下一帧
            if not infer_dataset.get_next_frame():
                print("End of dataset reached.")
                break
            
            # 判断是否有足够的历史帧进行推理
            has_enough_frames = infer_dataset.buffer.is_full()
            
            if not has_enough_frames:
                print(f"缓冲区未满 ({len(infer_dataset.buffer.color_imgs)}/{infer_dataset.buffer.buffer_size})，跳过处理...")
                frame_idx += 1
                continue
                
            # 缓冲区已满，准备进行推理
            print("准备推理数据...")
            color_img, pcloud, intrinsics, extrinsics, depth_target = infer_dataset.get_current_data()
            print(f"color_img shape: {color_img.shape}")
            print(f"pcloud shape: {pcloud.shape}")
            print(f"intrinsics shape: {intrinsics.shape}")
            print(f"extrinsics shape: {extrinsics.shape}")
            print(f"depth_target shape: {depth_target.shape}")
            
            # 添加以下代码 - 打印当前帧(最后一帧)的外参矩阵
            print("\n===== 当前帧外参矩阵 =====")
            # 获取当前帧(最后一帧)的外参矩阵
            current_ext = extrinsics[0, -1].numpy()  # 取批次0，最后一帧
            print(current_ext)

            # 可选：从外参矩阵中提取相机参数
            rotation_matrix = current_ext[:3, :3]
            translation = current_ext[:3, 3]

            # 计算相机高度
            camera_height = translation[2] * 1000  # 转换为毫米

            # 提取旋转角度
            inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            rot = R.from_matrix(inv_rotation_matrix)
            euler_angles = rot.as_euler('zyx', degrees=True)
            roll, pitch, yaw = euler_angles

            print(f"相机参数: 高度={camera_height:.2f}mm, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°, Roll={roll:.2f}°")
            print("========================\n")

            # 将数据移动到GPU
            color_img = color_img.to(device)
            pcloud = pcloud.to(device)  # 点云数据
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)
            depth_target = depth_target.to(device)

            # 执行推理
            print("Running inference...")
            trav_map, pred_depth, debug = model(color_img, pcloud, intrinsics, extrinsics, depth_target)
            
            # 先可视化结果
            save_path = os.path.join(vis_dir, f'frame_{frame_idx:06d}')
            visualize_results(trav_map, pred_depth, debug, color_img, depth_target, save_path=save_path)
            
            # 然后保存通过性图
            trav_map_np = trav_map.cpu().numpy()
            # 预处理：对每个通道进行转置和旋转，使其与可视化时的方向一致
            for i in range(trav_map_np.shape[1]):  # 遍历通道维度
                trav_map_np[0, i] = np.rot90(trav_map_np[0, i].T, k=1)  # 转置并逆时针旋转90度
            np.save(os.path.join(travmap_dir, f'trav_map_{frame_idx:06d}.npy'), trav_map_np)
            print(f"Saved traversability map to {os.path.join(travmap_dir, f'trav_map_{frame_idx:06d}.npy')}")
            
            # 保存最新的通过性图供MPPI使用
            np.save(os.path.join(travmap_dir, 'current_trav_map.npy'), trav_map_np)

            
            # 等待用户输入后继续
            user_input = input("\nPress Enter to continue, or 'q' to quit: ").lower()
            
            if user_input == 'q':
                print("Exiting program...")
                break
            
            frame_idx += 1
    
    print("Processing complete!")

if __name__ == "__main__":
    main()