import torch
import numpy as np
from lib.test.loss.attention_loss import visualize_attention

# 创建模拟的注意力权重
def create_dummy_attention_weights():
    """
    创建模拟的注意力权重
    """
    # 参数设置
    batch_size = 1
    num_heads = 4
    patch_size = 16
    image_size = 384
    num_patches = image_size // patch_size
    seq_len = num_patches * num_patches
    
    # 创建基础注意力权重
    # 中心区域注意力权重较高
    attn = np.zeros((batch_size, num_heads, seq_len, seq_len))
    
    # 计算中心补丁索引
    center_idx = (num_patches // 2) * num_patches + (num_patches // 2)
    
    # 为中心补丁添加较高的注意力权重
    for i in range(batch_size):
        for j in range(num_heads):
            # 中心补丁对自身的注意力
            attn[i, j, center_idx, center_idx] = 1.0
            
            # 中心周围的补丁
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_y = (num_patches // 2) + dy
                    neighbor_x = (num_patches // 2) + dx
                    if 0 <= neighbor_y < num_patches and 0 <= neighbor_x < num_patches:
                        neighbor_idx = neighbor_y * num_patches + neighbor_x
                        attn[i, j, center_idx, neighbor_idx] = 0.5
                        attn[i, j, neighbor_idx, center_idx] = 0.5
    
    # 转换为PyTorch张量
    attn_tensor = torch.tensor(attn, dtype=torch.float32)
    return attn_tensor, patch_size, image_size

if __name__ == "__main__":
    print("创建模拟注意力权重...")
    attn_weights, patch_size, image_size = create_dummy_attention_weights()
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"补丁大小: {patch_size}")
    print(f"图像大小: {image_size}")
    
    print("\n可视化注意力权重...")
    visualize_attention(attn_weights, patch_size, image_size)
    print("可视化完成！")
