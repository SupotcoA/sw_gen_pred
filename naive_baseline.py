import torch
import torch.nn.functional as F

def baseline_copy_mse_loss(reshaped_data, mask, s):
    """
    计算基线模型的MSE loss（使用重构后的数据）
    
    Args:
        reshaped_data: torch.Tensor, 形状为 [T, D*s] 的重构数据，其中 T = N/s
        mask: torch.Tensor, 形状为 [T, D*s] 的布尔掩码，True表示数据缺失
        s: int, 时间步长
    
    Returns:
        loss: torch.Tensor, MSE loss值
        valid_count: int, 有效预测的数量
    """
    T, Ds = reshaped_data.shape
    D = Ds // s
    
    # 分离输入和目标
    # 输入: 前T-1个时间步
    # 目标: 后T-1个时间步
    inputs = reshaped_data[:-1]  # [T-1, D*s]
    targets = reshaped_data[1:]  # [T-1, D*s]
    input_mask = mask[:-1]       # [T-1, D*s]
    target_mask = mask[1:]       # [T-1, D*s]
    
    # 基线预测: 用每个特征最后一个时间点的值预测接下来s个时间点
    # 从输入中提取每个特征最后一个时间点的值
    # 由于重构后的数据中，每个特征的时间序列被展开，我们需要提取每个特征块的最后一个值
    
    # 方法1: 高效向量化实现
    # 将输入数据重塑为 [T-1, D, s]，然后取最后一个时间点
    inputs_3d = inputs.reshape(T-1, D, s)  # [T-1, D, s]
    last_values = inputs_3d[:, :, -1]      # [T-1, D] 每个特征的最后一个值
    
    # 将最后一个值重复s次作为预测
    predictions = last_values.unsqueeze(-1).expand(-1, -1, s)  # [T-1, D, s]
    predictions = predictions.reshape(T-1, D*s)  # [T-1, D*s]
    
    # 处理缺失值
    # 1. 如果输入中某个特征的最后一个值缺失，则该特征的所有预测都无效
    # 2. 如果目标值缺失，则对应的预测无效
    
    # 检查输入中每个特征最后一个值是否缺失
    input_mask_3d = input_mask.reshape(T-1, D, s)  # [T-1, D, s]
    last_values_mask = input_mask_3d[:, :, -1]      # [T-1, D] 每个特征最后一个值的掩码
    
    # 如果输入最后一个值缺失，则该特征的所有s个预测都无效
    invalid_input_mask = last_values_mask.unsqueeze(-1).expand(-1, -1, s)  # [T-1, D, s]
    invalid_input_mask = invalid_input_mask.reshape(T-1, D*s)  # [T-1, D*s]
    
    # 最终的有效掩码：既不是输入缺失也不是目标缺失
    valid_mask = ~(invalid_input_mask | target_mask)  # [T-1, D*s]
    
    # 只计算有效位置的MSE
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    if valid_predictions.numel() == 0:
        # 如果没有有效预测，返回0
        return torch.tensor(0.0, device=reshaped_data.device), 0
    
    loss = F.mse_loss(valid_predictions, valid_targets, reduction='mean')
    
    return loss, valid_mask.sum().item()

def baseline_copy_mse_loss(reshaped_data, mask, s):
    """
    计算基线模型的MSE loss（使用重构后的数据）
    
    Args:
        reshaped_data: torch.Tensor, 形状为 [T, D*s] 的重构数据，其中 T = N/s
        mask: torch.Tensor, 形状为 [T, D*s] 的布尔掩码，True表示数据缺失
        s: int, 时间步长
    
    Returns:
        loss: torch.Tensor, MSE loss值
        valid_count: int, 有效预测的数量
    """
    T, Ds = reshaped_data.shape
    D = Ds // s
    
    # 分离输入和目标
    # 输入: 前T-1个时间步
    # 目标: 后T-1个时间步
    inputs = reshaped_data[:-1]  # [T-1, D*s]
    targets = reshaped_data[1:]  # [T-1, D*s]
    input_mask = mask[:-1]       # [T-1, D*s]
    target_mask = mask[1:]       # [T-1, D*s]
    
    # 基线预测: 用每个特征最后一个时间点的值预测接下来s个时间点
    # 从输入中提取每个特征最后一个时间点的值
    # 由于重构后的数据中，每个特征的时间序列被展开，我们需要提取每个特征块的最后一个值
    
    # 方法1: 高效向量化实现
    # 将输入数据重塑为 [T-1, D, s]，然后取最后一个时间点
    inputs_3d = inputs.reshape(T-1, D, s)  # [T-1, D, s]
    last_values = inputs_3d[:, :, -1]      # [T-1, D] 每个特征的最后一个值
    
    # 将最后一个值重复s次作为预测
    predictions = last_values.unsqueeze(-1).expand(-1, -1, s)  # [T-1, D, s]
    #predictions = predictions.reshape(T-1, D*s)  # [T-1, D*s]
    
    # 处理缺失值
    # 1. 如果输入中某个特征的最后一个值缺失，则该特征的所有预测都无效
    # 2. 如果目标值缺失，则对应的预测无效
    
    # 检查输入中每个特征最后一个值是否缺失
    input_mask_3d = input_mask.reshape(T-1, D, s)  # [T-1, D, s]
    last_values_mask = input_mask_3d[:, :, -1]      # [T-1, D] 每个特征最后一个值的掩码
    
    # 如果输入最后一个值缺失，则该特征的所有s个预测都无效
    invalid_input_mask = last_values_mask.unsqueeze(-1).expand(-1, -1, s)  # [T-1, D, s]
    #invalid_input_mask = invalid_input_mask.reshape(T-1, D*s)  # [T-1, D*s]
    
    # 最终的有效掩码：既不是输入缺失也不是目标缺失
    valid_mask = ~(invalid_input_mask | target_mask.reshape(T-1, D, s))  # [T-1, D, s]
    
    loss = (predictions-targets.reshape(T-1, D, s)).pow(2)
    loss[~valid_mask]=0
    loss=loss.sum(dim=(0,2))/(valid_mask.sum(dim=(0,2)).clamp(min=1))
    
    return loss, valid_mask.sum().item()

def baseline_copy_mean_mse_loss(reshaped_data, mask, s):
    """
    计算基线模型的MSE loss（使用重构后的数据）
    
    Args:
        reshaped_data: torch.Tensor, 形状为 [T, D*s] 的重构数据，其中 T = N/s
        mask: torch.Tensor, 形状为 [T, D*s] 的布尔掩码，True表示数据缺失
        s: int, 时间步长
    
    Returns:
        loss: torch.Tensor, MSE loss值
        valid_count: int, 有效预测的数量
    """
    T, Ds = reshaped_data.shape
    D = Ds // s
    
    # 分离输入和目标
    # 输入: 前T-1个时间步
    # 目标: 后T-1个时间步
    inputs = reshaped_data[:-1]  # [T-1, D*s]
    targets = reshaped_data[1:]  # [T-1, D*s]
    input_mask = mask[:-1]       # [T-1, D*s]
    target_mask = mask[1:]       # [T-1, D*s]
    
    # 基线预测: 用每个特征最后一个时间点的值预测接下来s个时间点
    # 从输入中提取每个特征最后一个时间点的值
    # 由于重构后的数据中，每个特征的时间序列被展开，我们需要提取每个特征块的最后一个值
    
    # 方法1: 高效向量化实现
    # 将输入数据重塑为 [T-1, D, s]，然后取最后一个时间点
    inputs_3d = inputs.reshape(T-1, D, s)  # [T-1, D, s]
    last_values = inputs_3d[:, :, -1]      # [T-1, D] 每个特征的最后一个值
    
    # 将最后一个值重复s次作为预测
    predictions = last_values.unsqueeze(-1).expand(-1, -1, s)  # [T-1, D, s]
    predictions = predictions.reshape(T-1, D*s)  # [T-1, D*s]
    
    # 处理缺失值
    # 1. 如果输入中某个特征的最后一个值缺失，则该特征的所有预测都无效
    # 2. 如果目标值缺失，则对应的预测无效
    
    # 检查输入中每个特征最后一个值是否缺失
    input_mask_3d = input_mask.reshape(T-1, D, s)  # [T-1, D, s]
    last_values_mask = input_mask_3d[:, :, -1]      # [T-1, D] 每个特征最后一个值的掩码
    
    # 如果输入最后一个值缺失，则该特征的所有s个预测都无效
    invalid_input_mask = last_values_mask.unsqueeze(-1).expand(-1, -1, s)  # [T-1, D, s]
    invalid_input_mask = invalid_input_mask.reshape(T-1, D*s)  # [T-1, D*s]
    
    # 最终的有效掩码：既不是输入缺失也不是目标缺失
    valid_mask = ~(invalid_input_mask | target_mask)  # [T-1, D*s]
    
    # 只计算有效位置的MSE
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    if valid_predictions.numel() == 0:
        # 如果没有有效预测，返回0
        return torch.tensor(0.0, device=reshaped_data.device), 0
    
    loss = F.mse_loss(valid_predictions, valid_targets, reduction='mean')
    
    return loss, valid_mask.sum().item()


if __name__ == "__main__":
    #test_baseline_mse_loss_reshaped()
    from data_seq import get_original_data
    @torch.no_grad()
    def preprocess(mask, x, seg_size=4):
        S, N = x.shape
        S=S//seg_size*seg_size
        x=x[:S]
        mask=mask[:S]

        x_reshaped = x.reshape(S//seg_size, seg_size, N)
        x_transformed = x_reshaped.permute(0, 2, 1).reshape(S//seg_size, N*seg_size)

        mask_reshaped = mask.reshape(S//seg_size, seg_size, N)
        mask_transformed = mask_reshaped.permute(0, 2, 1).reshape(S//seg_size, N*seg_size)
        return mask_transformed.contiguous() , x_transformed.contiguous()
    m, x=get_original_data("data/data") # [N,D]
    N=x.shape[0]
    s,e=int(N*0.8),int(N*0.9)
    m,x=m[s:e],x[s:e]
    # find the expected loss when always copy the last value (not token)
    seg_size=4
    m, x = preprocess(m,x,seg_size=seg_size) # [S//s,D*s]
    loss, valid = baseline_copy_mse_loss(x, m, seg_size)
    print(loss.mean()) #0.0454
    print(loss[3:7].mean()) # 0.0155
    print(loss)
    # [0.0765, 0.0850, 0.1632, 0.0171, 0.0033, 0.0226, 0.0193, 0.0187, 0.0025]


    #0.028267987072467804 sg=2
    #0.044439464807510376 sg=4 0.24932547668666885
    #multiplier=11.6865

    #1.7419807543434904 no prior