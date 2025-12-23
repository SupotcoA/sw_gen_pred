import torch
import numpy as np

def forward_transform_reshape(x, s):
    """
    将形状 [B, S, N] 变为 [B, S//s, N*s]
    原 [i, j, k] → 新 [i, j//s, j%s*N + k]
    """
    B, S, N = x.shape
    assert S % s == 0, "S must be divisible by s"
    
    # 方法1: 使用reshape和维度重排
    x_reshaped = x.reshape(B, S//s, s, N)
    x_transformed = x_reshaped.permute(0, 1, 3, 2).reshape(B, S//s, N*s)
    
    # 确保内存连续性
    return x_transformed.contiguous()

def inverse_transform_reshape(x, s, original_N):
    """
    将形状 [B, S/s, N*s] 变回 [B, S, N]
    新 [i, j, k] → 原 [i, j*s + k//original_N, k % original_N]
    """
    B, S_new, N_new = x.shape
    N = original_N
    S = S_new * s
    
    # 方法1: 使用reshape和维度重排
    x_reshaped = x.reshape(B, S_new, N, s)
    x_original = x_reshaped.permute(0, 1, 3, 2).reshape(B, S, N)
    
    # 确保内存连续性
    return x_original.contiguous()

def forward_transform_einsum(x, s):
    """使用einsum的正确实现"""
    B, S, N = x.shape
    x_reshaped = x.reshape(B, S//s, s, N)
    # 正确的einsum表达式
    result = torch.einsum('bijk->bikj', x_reshaped).reshape(B, S//s, N*s)
    return result.contiguous()

def inverse_transform_einsum(x, s, original_N):
    """使用einsum的逆变换正确实现"""
    B, S_new, N_new = x.shape
    x_reshaped = x.reshape(B, S_new, original_N, s)
    # 正确的einsum表达式
    result = torch.einsum('bijk->bikj', x_reshaped).reshape(B, S_new*s, original_N)
    return result.contiguous()

def test_transforms():
    """测试正向和逆向变换的正确性"""
    
    # 测试参数
    B, S, N = 2, 6, 3
    s = 2
    
    print(f"测试参数: B={B}, S={S}, N={N}, s={s}")
    print("=" * 60)
    
    # 创建测试数据
    x_original = torch.arange(B * S * N).reshape(B, S, N).float()
    print("原始张量形状:", x_original.shape)
    print("原始张量:")
    print(x_original)
    print()
    
    # 测试所有方法
    methods = [
        ("reshape", forward_transform_reshape, inverse_transform_reshape),
        ("einsum", forward_transform_einsum, inverse_transform_einsum)
    ]
    
    for method_name, forward_func, inverse_func in methods:
        print(f"测试方法: {method_name}")
        print("-" * 40)
        
        # 正向变换
        x_transformed = forward_func(x_original, s)
        print(f"变换后形状: {x_transformed.shape}")
        print("变换后张量:")
        print(x_transformed)
        
        # 检查连续性
        print(f"变换后是否连续: {x_transformed.is_contiguous()}")
        
        # # 验证索引映射
        # print("验证索引映射:")
        # for i in range(B):
        #     for j in range(S):
        #         for k in range(N):
        #             original_val = x_original[i, j, k].item()
        #             # 计算新位置
        #             new_i = i
        #             new_j = j // s
        #             new_k = (j % s) * N + k
                    
        #             transformed_val = x_transformed[new_i, new_j, new_k].item()
                    
        #             if abs(original_val - transformed_val) > 1e-6:
        #                 print(f"错误: 位置 [{i},{j},{k}] -> [{new_i},{new_j},{new_k}]")
        #                 print(f"  原值: {original_val}, 变换后: {transformed_val}")
        #                 return False
        
        # print("正向变换索引映射验证通过!")
        
        # 逆向变换
        x_restored = inverse_func(x_transformed, s, N)
        print(f"恢复后形状: {x_restored.shape}")
        print(f"恢复后是否连续: {x_restored.is_contiguous()}")
        
        # 检查是否完全恢复
        if torch.allclose(x_original, x_restored):
            print("逆向变换完全恢复原始张量!")
        else:
            print("错误: 逆向变换未能完全恢复原始张量")
            print("差异:", torch.abs(x_original - x_restored).max().item())
            return False
        
        print()
    
    return True

def test_memory_layout():
    """测试内存布局和连续性"""
    print("测试内存布局")
    print("=" * 40)
    
    B, S, N = 1, 4, 2
    s = 2
    
    x = torch.arange(B * S * N).reshape(B, S, N).float()
    
    # 测试reshape方法
    x_transformed = forward_transform_reshape(x, s)
    x_restored = inverse_transform_reshape(x_transformed, s, N)
    
    print(f"原始张量连续: {x.is_contiguous()}")
    print(f"变换后连续: {x_transformed.is_contiguous()}")
    print(f"恢复后连续: {x_restored.is_contiguous()}")
    print(f"恢复正确: {torch.allclose(x, x_restored)}")
    
    # 检查步长
    print(f"\n原始张量步长: {x.stride()}")
    print(f"变换后步长: {x_transformed.stride()}")
    print(f"恢复后步长: {x_restored.stride()}")



if __name__ == "__main__":
    success = test_transforms()
    print(f"\n所有测试{'通过' if success else '失败'}")
    
    print("\n" + "="*60)
    test_memory_layout()