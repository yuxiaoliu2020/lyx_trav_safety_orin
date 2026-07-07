import sys
import yaml
import copy
from easydict import EasyDict

_C = EasyDict()

_C.TAG = 'default'

_C.TRAINING = EasyDict()
_C.TRAINING.EPOCHS = 50             # 50 epochs
_C.TRAINING.BATCHSIZE = 4           # batch_size 为 4
_C.TRAINING.WORKERS = 8             # 使用 8 个工作线程进行数据加载
_C.TRAINING.DT = 0.1                # 时间步长为 0.1 秒
_C.TRAINING.HORIZON = 500           # 训练的时间窗为 500 步

_C.INFERENCE = EasyDict()
_C.INFERENCE.BATCHSIZE = 4           # batch_size 为 4
_C.INFERENCE.WORKERS = 8             # 使用 8 个工作线程进行数据加载

# 模型参数
_C.MODEL = EasyDict()
_C.MODEL.LOAD_NETWORK = "checkpoints/wayfaster.ckpt"  # 默认加载最后一个保存点
_C.MODEL.DOWNSAMPLE = 8             # 8 倍下采样
_C.MODEL.TIME_LENGTH = 3            # 时间长度为 3
_C.MODEL.LATENT_DIM = 64            # 隐变量维度为 64
_C.MODEL.PREDICT_DEPTH = True       # 默认要预测深度
_C.MODEL.FUSE_PCLOUD = True         # 默认要融合点云
_C.MODEL.INPUT_SIZE = (320, 180)    # 模型的输入为 320x180
_C.MODEL.GRID_BOUNDS = {            # 体素网格各个维度上的边界
    'xbound': [-2.0, 8.0, 0.1],     # x 轴边界为 [-2.0, 8.0]，步长为 0.1
    'ybound': [-5.0, 5.0, 0.1],     # y 轴边界为 [-5.0, 5.0]，步长为 0.1
    'zbound': [-1.0, 2.0, 0.1],     # z 轴边界为 [-1.0, 2.0]，步长为 0.1
    'dbound': [ 0.3, 8.0, 0.2]      # 深度轴 d 边界为 [0.3, 8.0]，步长为 0.2
}

_C.DATASET = EasyDict()
_C.DATASET.TEST_DATA = ['../dataset/zed2/data_valid', '../dataset/realsense/data_valid']
_C.DATASET.CSV_FILE = 'rosbags.csv'
_C.DATASET.INPUT_FOLDER = '/home/lyx/WildScenes_Dataset/offroad_dataset/sequence1'
_C.DATASET.OUTPUT_FOLDER = '/home/lyx/WildScenes_Dataset/offroad_dataset/sequence1_results'

# 随机种子为 42
_C.SEED = 42

# 合并两个配置文件
def merge_cfgs(base_cfg, new_cfg):
    config = copy.deepcopy(base_cfg)
    for key, val in new_cfg.items():
        if key in config:
            if type(config[key]) is EasyDict:
                config[key] = merge_cfgs(config[key], val)
            else:
                config[key] = val
        else:
            sys.exit("key {} doesn't exist in the default configs".format(key))

    return config


# 读取配置文件
def get_cfg(cfg_file):
    cfg = copy.deepcopy(_C)

    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = merge_cfgs(cfg, new_config)

    return cfg