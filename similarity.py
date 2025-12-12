import torch
import math


# ## Problem Statement

# We have two unknown conditional distributions \(P(x|z)\) and \(Q(x|z)\) for a scalar \(x\) given a high-dimensional condition \(z\). We can:
# - Sample \(z \sim p(z)\)
# - For each \(z\), draw **only one sample** from \(P(x|z)\) 
# - For each \(z\), draw **multiple samples** from \(Q(x|z)\)
# - Distributions are smooth in \(x\) but not necessarily in \(z\)

# **Goal**: Estimate \(\mathbb{E}_z[d(P(\cdot|z), Q(\cdot|z))]\) for some divergence \(d\).
# A bias term independent of Q is acceptable, as the goal is to compare different Qs.




# ## Proposed Methods

# ### 1. Cross-Entropy with Gaussian KDE

# **Estimator** for each \(z_i\):
# 1. Build KDE: \(\hat{q}_i(x) = \frac{1}{Mh}\sum_{j=1}^M \phi\left(\frac{x-y_{ij}}{h}\right)\) with Gaussian kernel \(\phi\)
# 2. Compute: \(\hat{d}_{\text{CE}}(z_i) = -\log \hat{q}_i(x_i)\)

# **Average**: \(\hat{D}_{\text{CE}} = \frac{1}{N}\sum_i \hat{d}_{\text{CE}}(z_i)\)

# **What it estimates**:
# \[
# \mathbb{E}_z[\underbrace{\text{KL}(P\|Q)}_{\text{Target}} + H(P) + \underbrace{\text{KDE bias}}_{\text{Depends on }M,h}]
# \]


# The following functions implements the above method, with vectorized calculations.

# The first cross_entropy_with_kde is the template.



@torch.no_grad()
def cross_entropy_with_kde(z, x0, mask, Q_func, reduce_dim=(0,2), NUM_SAMPLES_Q_PER_LOOP=4,**kwargs):
    """
    Estimates the similarity (Negative Log Likelihood) between P(x|z) and Q(x|z)
    element-wise across dimension s, aggregating over the batch b and dim.
    
    Args:
        z (torch.Tensor): Context [b, s, cond_dim]
        x0 (torch.Tensor): One sample from P(x|z) [b, s, dim]
        mask (torch.Tensor): Boolean mask [b, s, dim]. True indicates x0 is NaN/Invalid.
        Q_func (callable): Function where Q_func(z) returns samples [b, s, dim].
        reduce_dim (list | tuple): Which dim in [b, s, dim] is averaged in the output.
        
    Returns:
        mean_nll (torch.Tensor): Mean NLL per (s,) [s,] (depending on reduce_dim)
        std_nll (torch.Tensor): Std of NLL per (s,) [s,] (depending on reduce_dim)
    """
    # --- Hyperparameters ---
    NUM_SAMPLES_Q = 64
    MIN_STD = 1e-2 
    
    b, s, dim = x0.shape
    
    # 1. Sampling from Q
    # We repeat z to draw multiple samples in parallel.
    # z.repeat(M, 1, 1) results in shape [M*b, s, cond_dim] 
    # with ordering [b_0, b_1, ..., b_0, b_1, ...]
    # 使用分块处理减少显存占用
    NUM_LOOPS = NUM_SAMPLES_Q // NUM_SAMPLES_Q_PER_LOOP
    
    # 用于存储所有块的采样结果
    samples_q_list = []
    
    # 分块处理，每次只处理一部分采样
    for i in range(NUM_LOOPS):
        # 重复当前块所需的采样数
        z_expanded = z.repeat(NUM_SAMPLES_Q_PER_LOOP, 1, 1)
        
        # raw_samples shape: [M_per_loop * b, s, dim]
        raw_samples = Q_func(z_expanded, **kwargs)
        
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
    q_std = samples_q.std(dim=0, unbiased=False)
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
    
    # Count valid entries per (s,)
    valid_counts = valid_mask.sum(dim=reduce_dim).float()
    
    # Avoid division by zero for empty slices (clamp min count to 1.0)
    # If a slice is fully NaN, result will be 0.0 (which is arbitrary but safe)
    safe_counts = torch.clamp(valid_counts, min=1.0)
    
    # Calculate Mean
    mean_val = nll_clean.sum(dim=reduce_dim) / safe_counts
    
    # Calculate Standard Deviation
    # We calculate variance using the "sum of squared differences" method
    mean_expanded = mean_val
    for axis in sorted(reduce_dim):
        mean_expanded = mean_expanded.unsqueeze(axis)

    diff_from_mean = nll_clean - mean_expanded
    sq_diff = diff_from_mean ** 2
    
    # Mask out invalid squared differences
    sq_diff_clean = sq_diff.masked_fill(mask, 0.0)
    
    # Use (N-1) for unbiased std, but clamp to 1.0 to avoid error
    unbiased_counts = torch.clamp(valid_counts - 1, min=1.0)
    
    var_val = sq_diff_clean.sum(dim=reduce_dim) / unbiased_counts
    std_val = torch.sqrt(var_val)
    
    return mean_val, std_val


# """
# ### 2. MMD with Characteristic Kernels

# **Estimator** for each \(z_i\):
# \[
# \hat{d}_k(z_i) = k(x_i, x_i) + \frac{1}{M^2}\sum_{j,k} k(y_{ij}, y_{ik}) - \frac{2}{M}\sum_j k(x_i, y_{ij})
# \]

# **Average**: \(\hat{D} = \frac{1}{N}\sum_i \hat{d}_k(z_i)\)

# **What it estimates**: 
# \[
# \mathbb{E}_z[\text{MMD}^2(P,Q) + \underbrace{\mathbb{E}_{x\sim P}[k(x,x)] - \mathbb{E}_{x,x'\sim P}[k(x,x')]}_{\text{Bias independent of }Q}]
# \]

# **Kernel choices**:
# - **Gaussian RBF**: \(k(x,y)=\exp(-|x-y|^2/(2\sigma^2))\)
# - **Laplace**: \(k(x,y)=\exp(-|x-y|/\sigma)\)
# - **Bandwidth selection**: to be considered
# """

@torch.no_grad()
def mmd(z, x0, mask, Q_func, reduce_dim=(0,2), NUM_SAMPLES_Q_PER_LOOP=4,**kwargs):
    """
    Vectorized MMD^2 estimator per-element (treating each scalar in last dim independently).

    Follows the same input/output and aggregation semantics as cross_entropy_with_kde.
    Uses an RBF kernel with a bandwidth computed from the sample std (Scott-like rule).
    Returns (mean_mmd, std_mmd) aggregated across dimensions given by reduce_dim.
    """
    # Hyperparameters (kept consistent with KDE function)
    NUM_SAMPLES_Q = 64
    MIN_STD = 1e-2
    NUM_LOOPS = NUM_SAMPLES_Q // NUM_SAMPLES_Q_PER_LOOP

    b, s, dim = x0.shape

    # 1. Sample from Q in blocks (same strategy as cross_entropy_with_kde)
    samples_q_list = []
    for i in range(NUM_LOOPS):
        z_expanded = z.repeat(NUM_SAMPLES_Q_PER_LOOP, 1, 1)
        raw_samples = Q_func(z_expanded, **kwargs)  # expected shape: [M_per_loop * b, s, dim]
        samples_q_per_loop = raw_samples.view(
            NUM_SAMPLES_Q_PER_LOOP, b, s, dim
        )  # [M_per_loop, b, s, dim]
        samples_q_list.append(samples_q_per_loop)

    samples_q = torch.cat(samples_q_list, dim=0)  # [M, b, s, dim]
    M = samples_q.shape[0]

    # 2. Treat every scalar element (b, s, dim) as independent 1D tasks
    # Flatten tasks axis: N = b * s * dim
    samples_flat = samples_q.view(M, -1)  # [M, N]
    x0_flat = x0.view(-1)  # [N]
    mask_flat = mask.view(-1)  # [N] boolean

    # Bandwidth per task using Scott-like rule from sample std (over M)
    q_std = samples_q.std(dim=0, unbiased=False)  # [b, s, dim]
    flat_std = q_std.view(-1)  # [N]
    bandwidth = 1.06 * flat_std * (M ** -0.2)
    bandwidth = torch.clamp(bandwidth, min=MIN_STD)  # [N]
    bw_sq = (bandwidth ** 2).unsqueeze(0)  # [1, N] for broadcasting

    # 3. Compute pairwise kernel among samples: term2 = (1/M^2) sum_{i,j} k(y_i, y_j)

    # Compute it in an M-loop to avoid allocating [M, M, N] (huge!). Use in-place ops.
    # samples_flat: [M, N], bw_sq: [1, N]
    term2_flat = torch.zeros_like(x0_flat)  # [N]
    den = 2.0 * bw_sq  # [1, N]

    # Loop over i, compute k(y_i, y_j) for all j producing [M, N] temporaries (much smaller)
    for i in range(M):
        # diff: [M, N] = y_j - y_i
        diff = samples_flat - samples_flat[i : i + 1]  # new tensor [M, N]
        # square, scale and exponentiate in-place to produce kernel entries for this i
        diff.mul_(diff)           # diff = (y_j - y_i)^2
        diff.div_(den)            # diff = dist2 / (2 * bw_sq)
        diff.neg_().exp_()        # diff = exp(-dist2/(2*bw_sq))
        term2_flat.add_(diff.sum(dim=0))  # accumulate sum_j k(y_i, y_j) for each task

    term2_flat = term2_flat / float(M * M)  # [N]

    # # Compute pairwise squared differences per task using broadcasting
    # # diff shape -> [M, M, N]
    # diff_pairs = samples_flat.unsqueeze(0) - samples_flat.unsqueeze(1)
    # dist2_pairs = diff_pairs.pow(2)  # scalar tasks -> squared differences
    # K_pairs = torch.exp(-dist2_pairs / (2.0 * bw_sq))  # [M, M, N]
    # term2_flat = K_pairs.sum(dim=(0, 1)) / float(M * M)  # [N]

    # 4. Compute cross term: term3 = (2/M) sum_j k(x0, y_j)
    diff_x = samples_flat - x0_flat.unsqueeze(0)  # [M, N]
    dist2_x = diff_x.pow(2)  # [M, N]
    K_x = torch.exp(-dist2_x / (2.0 * bw_sq))  # [M, N]
    term3_flat = 2.0 * K_x.sum(dim=0) / float(M)  # [N]

    # 5. k(x,x) = 1 (RBF at zero distance)
    k_xx_flat = torch.ones_like(term2_flat)

    mmd2_flat = k_xx_flat + term2_flat - term3_flat  # [N]

    # 6. Restore shape [b, s, dim]
    mmd2 = mmd2_flat.view(b, s, dim)

    # 7. Aggregation with masking (same semantics as cross_entropy_with_kde)
    valid_mask = ~mask  # True where data is GOOD
    mmd_clean = mmd2.masked_fill(mask, 0.0)

    # Count valid entries per reduce_dim
    valid_counts = valid_mask.sum(dim=reduce_dim).float()
    safe_counts = torch.clamp(valid_counts, min=1.0)

    mean_val = mmd_clean.sum(dim=reduce_dim) / safe_counts

    # Compute standard deviation (unbiased with N-1 clamped to >=1)
    # Build mean expanded to original shape for subtraction
    mean_expanded = mean_val
    for axis in sorted(reduce_dim):
        mean_expanded = mean_expanded.unsqueeze(axis)

    diff_from_mean = mmd_clean - mean_expanded
    sq_diff = diff_from_mean.pow(2)
    sq_diff_clean = sq_diff.masked_fill(mask, 0.0)

    unbiased_counts = torch.clamp(valid_counts - 1.0, min=1.0)
    var_val = sq_diff_clean.sum(dim=reduce_dim) / unbiased_counts
    std_val = torch.sqrt(var_val)

    return mean_val, std_val


# """
# ### 3. Probability Integral Transform (PIT) Approach

# ### Method
# For each condition \(z_i \sim p(z)\):
# 1. Draw one sample \(x_i \sim P(x|z_i)\)
# 2. Draw \(M\) samples \(\{y_{ij}\}_{j=1}^M \sim Q(x|z_i)\)
# 3. Compute percentile: \(\hat{u}_i = \frac{\text{rank}(x_i \text{ among } \{x_i, y_{i1},..., y_{iM}\})}{M+1}\)

# Collect \(\{\hat{u}_i\}_{i=1}^N\) and measure deviation from Uniform(0,1) using **Cramér-von Mises statistic**:
# \[
# D_{\text{CvM}} = \frac{1}{12N} + \sum_{i=1}^N \left(\hat{u}_{(i)} - \frac{2i-1}{2N}\right)^2
# \]
# where \(\hat{u}_{(i)}\) are sorted percentiles.

# ### What It Estimates
# Measures **average calibration error** over \(z\):
# - If \(P = Q\) conditionally for all \(z\), \(\{\hat{u}_i\} \sim \text{Uniform}(0,1)\)
# - Large \(D_{\text{CvM}}\) indicates miscalibration

# ### Advantages
# 1. **No tuning parameters** (except \(M\), no bandwidth selection)
# 2. **Interpretable**: Directly measures probability coverage
# 3. **Scale-invariant**: Works under any monotonic transformation of \(x\)
# 4. **Detects systematic biases**: Skewed percentiles reveal over/underestimation

# ### Shortcomings
# 1. **Marginal (not conditional) calibration**: Tests average over \(z\), may miss \(z\)-dependent errors
# 2. **May miss higher-moment differences**: Two distributions with same average CDF but different conditional shapes could appear equal
# 3. **One-sided**: Optimizes for calibration, not necessarily all distributional aspects

# ### Best Use Case
# Comparing \(Q_1\) vs \(Q_2\) using same \(\{z_i, x_i\}\):  
# Smaller \(D_{\text{CvM}}\) indicates better average calibration relative to \(P\).
# """

@torch.no_grad()
def pit_cvm(z, x0, mask, Q_func, reduce_dim=(0,2), NUM_SAMPLES_Q_PER_LOOP=4,**kwargs):
    """
    Probability Integral Transform (PIT) using Cramér-von Mises statistic.

    Follows the same input/output and aggregation semantics as cross_entropy_with_kde:
    - z: [b, s, cond_dim]
    - x0: [b, s, dim] one sample from P per (b,s,dim)
    - mask: [b, s, dim] boolean, True indicates invalid x0
    - Q_func: callable(z_expanded, **kwargs) -> samples [M_per_loop * b, s, dim]
    - reduce_dim: tuple of dims to aggregate over (e.g. (0,2) to keep s dimension)

    Returns:
        mean_cvm (torch.Tensor): D_CvM per kept-slice (shape depends on reduce_dim)
        std_cvm (torch.Tensor): Std of per-element contributions to the CvM sum per slice
    """
    # Hyperparameters (consistent with other functions)
    NUM_SAMPLES_Q = 64
    NUM_LOOPS = NUM_SAMPLES_Q // NUM_SAMPLES_Q_PER_LOOP

    b, s, dim = x0.shape

    # 1. Sample from Q in blocks (same strategy as other functions)
    samples_q_list = []
    for i in range(NUM_LOOPS):
        z_expanded = z.repeat(NUM_SAMPLES_Q_PER_LOOP, 1, 1)
        raw_samples = Q_func(z_expanded, **kwargs)  # expected shape: [M_per_loop * b, s, dim]
        samples_q_per_loop = raw_samples.view(
            NUM_SAMPLES_Q_PER_LOOP, b, s, dim
        )  # [M_per_loop, b, s, dim]
        samples_q_list.append(samples_q_per_loop)

    samples_q = torch.cat(samples_q_list, dim=0)  # [M, b, s, dim]
    M = samples_q.shape[0]

    # 2. Compute empirical CDF value (percentile) for each (b,s,dim) task:
    # rank = 1 + number of y_j <= x0  -> percentile = rank / (M+1)
    x0_expanded = x0.unsqueeze(0)  # [1, b, s, dim]
    le_counts = (samples_q <= x0_expanded).sum(dim=0).to(dtype=torch.float32)  # [b, s, dim]
    ranks = le_counts + 1.0  # rank in {1,...,M+1}
    u = ranks / float(M + 1)  # empirical percentile in (0,1]

    # 3. Prepare grouping for reduction specified by reduce_dim
    shape = list(u.shape)  # [b, s, dim]
    dims = len(shape)
    reduce_dims = tuple(sorted(reduce_dim))
    keep_dims = [i for i in range(dims) if i not in reduce_dims]

    # Permute so reduce dims come first, keep dims last
    perm = list(reduce_dims) + keep_dims
    u_perm = u.permute(perm)  # shape: [*reduce_sizes, *keep_sizes]
    valid_mask = (~mask).permute(perm)  # True where valid

    reduce_sizes = [shape[d] for d in reduce_dims] if reduce_dims else [1]
    keep_sizes = [shape[d] for d in keep_dims] if keep_dims else [1]

    R = int(math.prod(reduce_sizes))  # number of items to aggregate over per kept-slice
    K = int(math.prod(keep_sizes))    # number of kept-slices (e.g., s)

    u_groups = u_perm.reshape(R, K)  # [R, K]
    valid_groups = valid_mask.reshape(R, K)  # [R, K]

    device = u.device
    dtype = u.dtype

    mean_out = torch.zeros(K, device=device, dtype=dtype)
    std_out = torch.zeros(K, device=device, dtype=dtype)

    # 4. For each kept-slice (column), compute CvM statistic over valid u's
    for k in range(K):
        vm = valid_groups[:, k]
        if vm.sum() == 0:
            mean_out[k] = 0.0
            std_out[k] = 0.0
            continue

        u_valid = u_groups[vm, k].to(dtype=torch.float64)  # use double for sorting / numerics
        N = u_valid.numel()

        # sort percentiles
        u_sorted, _ = torch.sort(u_valid)

        # expected uniform order statistics: (2i-1)/(2N), i=1..N
        idx = torch.arange(1, N + 1, device=device, dtype=torch.float64)
        expected = (2.0 * idx - 1.0) / (2.0 * float(N))

        # per-element contributions
        contributions = (u_sorted - expected).pow(2)  # length N

        # CvM statistic
        D_cvm = (1.0 / (12.0 * float(N))) + contributions.sum()

        # store mean (the statistic) and std (std of contributions)
        mean_out[k] = D_cvm.to(dtype)
        if N > 1:
            var_c = contributions.var(unbiased=True)
            std_out[k] = torch.sqrt(var_c).to(dtype)
        else:
            std_out[k] = 0.0

    # 5. Restore keep_dims shape
    out_shape = keep_sizes if keep_sizes != [1] else [1]
    mean_val = mean_out.reshape(*out_shape)
    std_val = std_out.reshape(*out_shape)

    # If keep_dims was empty (we reduced all dims), return scalars
    if not keep_dims:
        mean_val = mean_val.squeeze()
        std_val = std_val.squeeze()

    return mean_val, std_val

