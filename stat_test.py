import math

def find_sample_size_hoeffding(a, delta):
    """
    使用 Hoeffding 界计算达到精度 a 所需的最小采样次数
    
    参数:
        a: 绝对误差容限（百分位数的绝对误差，如0.05表示5%）
        delta: 置信水平参数（1-delta为置信水平，如0.05表示95%置信）
    
    返回:
        所需的最小采样次数 n
    """
    n = 1  # 从1开始尝试
    
    while True:
        # 计算 Hoeffding 半径
        r_n = math.sqrt((1/(2*n)) * math.log((2 * math.pi**2 * n**2) / (6 * delta)))
        
        # 如果达到精度要求，返回 n
        if r_n <= a:
            return n
        
        n += 1
        
        # 设置一个安全上限，避免无限循环
        if n > 10000000:  # 1000万次采样上限
            return n

# 设置参数
a = 0.05  # 5% 绝对误差
delta = 0.5  # 95% 置信水平

# 计算所需采样次数
required_n = find_sample_size_hoeffding(a, delta)
print(f"使用 Hoeffding 界，在 a={a}, delta={delta} 时，所需采样次数约为: {required_n:,}")

# 验证结果
r_n = math.sqrt((1/(2*required_n)) * math.log((2 * math.pi**2 * required_n**2) / (6 * delta)))
print(f"验证: 当 n={required_n} 时，r_n = {r_n:.6f} (≤ {a})")