import torch
import math

@torch.no_grad()
def evaluate_conditional_similarity(z, x0, mask, Q_func):
    """
    Estimates the similarity (Negative Log Likelihood) between P(x|z) and Q(x|z)
    element-wise across dimensions s and dim, aggregating over the batch b.
    
    Args:
        z (torch.Tensor): Context [b, s, cond_dim]
        x0 (torch.Tensor): One sample from P(x|z) [b, s, dim]
        mask (torch.Tensor): Boolean mask [b, s, dim]. True indicates x0 is NaN/Invalid.
        Q_func (callable): Function where Q_func(z) returns samples [b, s, dim].
        
    Returns:
        mean_nll (torch.Tensor): Mean NLL per (s, dim) [s, dim]
        std_nll (torch.Tensor): Std of NLL per (s, dim) [s, dim]
    """
    # --- Hyperparameters ---
    NUM_SAMPLES_Q = 64
    MIN_STD = 1e-4 
    
    b, s, dim = x0.shape
    
    # 1. Sampling from Q
    # We repeat z to draw multiple samples in parallel.
        # z.repeat(M, 1, 1) results in shape [M*b, s, cond_dim] 
    # with ordering [b_0, b_1, ..., b_0, b_1, ...]
    # 使用分块处理减少显存占用
    NUM_SAMPLES_Q_PER_LOOP = 4
    NUM_LOOPS = NUM_SAMPLES_Q // NUM_SAMPLES_Q_PER_LOOP
    
    # 用于存储所有块的采样结果
    samples_q_list = []
    
    # 分块处理，每次只处理一部分采样
    for i in range(NUM_LOOPS):
        # 重复当前块所需的采样数
        z_expanded = z.repeat(NUM_SAMPLES_Q_PER_LOOP, 1, 1)
        
        # raw_samples shape: [M_per_loop * b, s, dim]
        raw_samples = Q_func(z_expanded)
        
        # Reshape to [M_per_loop, b, s, dim] 隔离每个z_i的M_per_loop个采样
        samples_q_per_loop = raw_samples.view(
            NUM_SAMPLES_Q_PER_LOOP, b, s, dim
        )
        
        samples_q_list.append(samples_q_per_loop)
    
    # 沿着第一个维度(M维度)拼接所有块的采样结果
    # 最终形状为[NUM_SAMPLES_Q, b, s, dim]
    samples_q = torch.cat(samples_q_list, dim=0)
    
    # 2. Vectorized 1D Kernel Density Estimation
    # We treat every element (b, s, dim) as an independent 1D distribution estimation task.
    
    # Calculate Bandwidth using Scott's Rule: h = 1.06 * sigma * M^(-1/5)
    # Std is calculated over the M dimension (dim=0)
    q_std = samples_q.std(dim=0)
    q_std = torch.clamp(q_std, min=MIN_STD)
    bandwidth = 1.06 * q_std * (NUM_SAMPLES_Q ** -0.2)
    
    # Prepare tensors for broadcasting
    # samples_q: [M, b, s, dim]
    # x0: [1, b, s, dim]
    # bandwidth: [1, b, s, dim]
    x0_expanded = x0.unsqueeze(0)
    bandwidth_expanded = bandwidth.unsqueeze(0)
    
    # Gaussian Kernel calculation
    # We work in log-space for numerical stability (Log-Sum-Exp trick)
    diff = samples_q - x0_expanded
    log_exponent = -0.5 * (diff / bandwidth_expanded) ** 2
    
    # Normalization constant for 1D Gaussian: 1 / (h * sqrt(2*pi))
    # Log norm: -log(h) - 0.5 * log(2*pi)
    log_norm = -torch.log(bandwidth_expanded) - 0.5 * math.log(2 * math.pi)
    
    log_kernels = log_exponent + log_norm  # Shape: [M, b, s, dim]
    
    # 3. Compute Log-Likelihood of x0 under Q
    # Density = (1/M) * sum( exp(log_kernels) )
    # LogDensity = -log(M) + LogSumExp(log_kernels)
    log_sum_exp = torch.logsumexp(log_kernels, dim=0) # Reduces M dimension -> [b, s, dim]
    log_prob = -math.log(NUM_SAMPLES_Q) + log_sum_exp
    
    # NLL is our similarity metric (lower is better, i.e., closer distributions)
    nll = -log_prob
    
    # 4. Aggregation with Masking
    # valid_mask is True where data is GOOD.
    valid_mask = ~mask 
    
    # Zero out invalid NLLs so they don't affect the sum
    # (Note: nll may contain NaNs where x0 was NaN, so we must fill them)
    nll_clean = nll.masked_fill(mask, 0.0)
    
    # Count valid entries per (s, dim)
    valid_counts = valid_mask.sum(dim=0).float()
    
    # Avoid division by zero for empty slices (clamp min count to 1.0)
    # If a slice is fully NaN, result will be 0.0 (which is arbitrary but safe)
    safe_counts = torch.clamp(valid_counts, min=1.0)
    
    # Calculate Mean
    mean_val = nll_clean.sum(dim=0) / safe_counts
    
    # Calculate Standard Deviation
    # We calculate variance using the "sum of squared differences" method
    diff_from_mean = nll - mean_val.unsqueeze(0)
    sq_diff = diff_from_mean ** 2
    
    # Mask out invalid squared differences
    sq_diff_clean = sq_diff.masked_fill(mask, 0.0)
    
    # Use (N-1) for unbiased std, but clamp to 1.0 to avoid error
    unbiased_counts = torch.clamp(valid_counts - 1, min=1.0)
    
    var_val = sq_diff_clean.sum(dim=0) / unbiased_counts
    std_val = torch.sqrt(var_val)
    
    return mean_val, std_val