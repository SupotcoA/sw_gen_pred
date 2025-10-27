import torch
import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

def load_mat_file(root):
    data=dict()
    with h5py.File(os.path.join(root,"ACE_2012.mat"),"r") as f:
        for key in f.keys():
            data[key]=np.asarray(f[key]).transpose()
    with h5py.File(os.path.join(root,"OMNI_2012.mat"),"r") as f:
        for key in f.keys():
            data[key]=np.asarray(f[key]).transpose()
    return data

def preprocess_data(data:dict):
    key = 'ACE_Bz_2012'
    arr = data[key]
    arr[arr<-1e3]=0.0
    arr /= 3.735161
    data[key] = arr.astype(np.float32)

    key='ACE_Psw_2012'
    arr = data[key]
    arr /= 3148
    data[key] = arr.astype(np.float32)

    key='ACE_Vsw_2012'
    arr = data[key]
    arr = (arr - 383)/341
    data[key] = arr.astype(np.float32)

    key='OMNI_AE_2012'
    arr = data[key]
    arr = (arr - 177)/212
    data[key] = arr.astype(np.float32)

    key='OMNI_ASYMH_2012'
    arr = data[key]
    arr = (arr - 19) / 14 #
    data[key] = arr.astype(np.float32)
    
    key='OMNI_PC_2012'
    arr = data[key]
    arr[arr>100]=1.0
    arr = (arr - 1) / 1.465 #
    data[key] = arr.astype(np.float32)

    key='OMNI_SYMH_2012'
    arr = data[key]
    arr = (arr + 11) / 18.755921 
    data[key] = arr.astype(np.float32)

    print("Using data:",*data.keys())

    data_array = [torch.from_numpy(data[key]).float() for key in data.keys()]
    data_array = torch.cat(data_array,dim=1)
    print("All data cat shape:", data_array.shape)
    return data_array

@torch.no_grad()
def get_original_data(root):
    return preprocess_data(load_mat_file(root))

    

class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, split='train', split_ratios=[0.5, 0.25, 0.25]):
        """
        多变量时间序列数据集
        
        Args:
            data: 原始数据，形状为 (N, dim)
            seq_len: 序列长度
            split: 数据集类型 ('train', 'val', 'test')
            split_ratios: 训练/验证/测试集划分比例
        """
        self.data = data
        self.seq_len = seq_len
        self.split = split
        
        # 按比例划分数据
        n_total = len(data)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        if split == 'train':
            self.data_segment = data[:n_train]
            print("train data len", self.data_segment.shape[0])
        elif split == 'val':
            self.data_segment = data[n_train:n_train + n_val]
            print("val data len", self.data_segment.shape[0])
        elif split == 'test':
            self.data_segment = data[n_train + n_val:]
            print("test data len", self.data_segment.shape[0])
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")
        
        # 计算可用样本数
        self.n_samples = len(self.data_segment) - seq_len + 1
        
        # 如果样本数不足，抛出错误
        if self.n_samples <= 0:
            raise ValueError(f"序列长度 {seq_len} 大于{split}集数据长度 {len(self.data_segment)}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 从指定位置开始获取seq_len长度的序列
        sequence = self.data_segment[idx:idx + self.seq_len]
        return sequence

class RandomMultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, split='train', split_ratios=[0.5, 0.25, 0.25], num_samples=None):
        """
        随机采样的多变量时间序列数据集
        
        Args:
            data: 原始数据，形状为 (N, dim)
            seq_len: 序列长度
            split: 数据集类型 ('train', 'val', 'test')
            split_ratios: 训练/验证/测试集划分比例
            num_samples: 样本数量，如果为None则自动计算
        """
        self.data = data
        self.seq_len = seq_len
        self.split = split
        
        # 按比例划分数据
        n_total = len(data)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        if split == 'train':
            self.data_segment = data[:n_train]
            print("train data len", self.data_segment.shape[0])
        elif split == 'val':
            self.data_segment = data[n_train:n_train + n_val]
            print("val data len", self.data_segment.shape[0])
        elif split == 'test':
            self.data_segment = data[n_train + n_val:]
            print("test data len", self.data_segment.shape[0])
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")
        
        # 设置样本数量
        if num_samples is None:
            # 默认样本数量为数据段长度的5倍
            self.num_samples = len(self.data_segment) * 5
        else:
            self.num_samples = num_samples
        
        # 计算可能的起始位置范围
        self.valid_start_indices = len(self.data_segment) - seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 随机选择起始位置
        start_idx = torch.randint(0, self.valid_start_indices + 1, (1,)).item()
        
        # 获取序列
        sequence = self.data_segment[start_idx:start_idx + self.seq_len]
        return sequence

def prepare_datasets(data_config):
    """
    准备时间序列数据集
    
    Args:
        data_config: 数据配置字典
            shape: (batch_size, max_seq_len, inp_dim)
            batch_size: 批量大小
            split: 训练/验证/测试集划分比例
            space_weather_data_root: 数据路径（如果get_original_data需要）
    
    Returns:
        train_dataset, val_dataset, test_dataset: 数据集对象
    """
    # 获取原始数据
    S = get_original_data(data_config['space_weather_data_root'])  # 形状 (N, dim)
    
    # 从配置中提取参数
    batch_size, max_seq_len, inp_dim = data_config['shape']
    assert inp_dim == S.shape[-1], f"data dim {inp_dim}; expected shape {S.shape}"
    split_ratios = data_config['split']

    max_seq_len += 1 # TODO: consider this
    
    # 创建数据集
    train_dataset = RandomMultivariateTimeSeriesDataset(
        data=S,
        seq_len=max_seq_len,
        split='train',
        split_ratios=split_ratios,
        num_samples=None
    )
    
    val_dataset = MultivariateTimeSeriesDataset(
        data=S,
        seq_len=max_seq_len,
        split='val',
        split_ratios=split_ratios
    )
    
    test_dataset = MultivariateTimeSeriesDataset(
        data=S,
        seq_len=max_seq_len,
        split='test',
        split_ratios=split_ratios
    )
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(data_config):
    """
    创建数据加载器
    
    Args:
        data_config: 数据配置字典
    
    Returns:
        train_loader, val_loader, test_loader: 数据加载器
    """
    train_dataset, val_dataset, test_dataset = prepare_datasets(data_config)
    
    batch_size = data_config['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        #pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        #pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 假设的数据配置
    data_config = dict(
        shape=(32, 100, 8),  # batch_size=32, max_seq_len=100, inp_dim=10
        batch_size=32,
        split=[0.5, 0.25, 0.25],
        space_weather_data_root="data"
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(data_config)
    
    # 使用方式
    for x0 in train_loader:
        x0 = x0.to('cuda')  # 假设模型在GPU上
        print(x0.dtype)
        # x0的形状: (batch_size, max_seq_len, dim)
        print(f"Batch shape: {x0.shape}")
        break  # 只演示第一个batch
