import torch
import torch.nn as nn
import numpy as np


def calculate_bbox_patch_indices(bbox, patch_size, image_size):
    """
    计算边界框覆盖的补丁索引
    
    Args:
        bbox: 边界框坐标 (x, y, w, h)
        patch_size: 补丁大小
        image_size: 图像大小
    
    Returns:
        边界框覆盖的补丁索引列表
    """
    x, y, w, h = bbox
    # 计算边界框的像素坐标范围
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # 计算补丁的行列数
    num_patches = image_size // patch_size
    
    # 计算边界框覆盖的补丁索引
    patch_indices = []
    for i in range(max(0, y1 // patch_size), min(num_patches, (y2 + patch_size - 1) // patch_size)):
        for j in range(max(0, x1 // patch_size), min(num_patches, (x2 + patch_size - 1) // patch_size)):
            patch_idx = i * num_patches + j
            patch_indices.append(patch_idx)
    
    return patch_indices


class EncoderAttentionLoss(nn.Module):
    """
    编码器注意力损失函数
    目标：降低正确边界框对应的补丁的注意力权重
    """
    def __init__(self, patch_size=16, search_size=384):
        super().__init__()
        self.patch_size = patch_size
        self.search_size = search_size
    
    def forward(self, attn_weights, bbox):
        """
        计算编码器注意力损失
        
        Args:
            attn_weights: 编码器注意力权重列表，每个元素形状为 [batch, num_heads, seq_len, seq_len]
            bbox: 正确的边界框坐标 (x, y, w, h)
        
        Returns:
            损失值
        """
        if not attn_weights:
            return 0.0
        
        # 计算正确边界框对应的补丁索引
        patch_indices = calculate_bbox_patch_indices(bbox, self.patch_size, self.search_size)
        
        if not patch_indices:
            return 0.0
        
        total_loss = 0.0
        for layer_attn in attn_weights:
            # 只关注搜索区域内部的注意力
            # 假设序列的前半部分是搜索区域，后半部分是模板区域
            search_seq_len = (self.search_size // self.patch_size) ** 2
            
            # 提取搜索区域内部的注意力权重
            search_attn = layer_attn[:, :, :search_seq_len, :search_seq_len]
            
            # 计算正确补丁的平均注意力权重
            correct_attn = search_attn[:, :, :, patch_indices].mean(dim=-1)
            
            # 损失：降低这些权重
            loss = correct_attn.mean()
            total_loss += loss
        
        return total_loss / len(attn_weights)


class DecoderAttentionLoss(nn.Module):
    """
    解码器注意力损失函数
    目标：降低解码器对正确边界框对应特征的注意力权重
    """
    def __init__(self, patch_size=16, search_size=384):
        super().__init__()
        self.patch_size = patch_size
        self.search_size = search_size
    
    def forward(self, cross_attn_weights, bbox):
        """
        计算解码器交叉注意力损失
        
        Args:
            cross_attn_weights: 解码器交叉注意力权重列表，每个元素是一个列表，包含每一层的注意力权重
            bbox: 正确的边界框坐标 (x, y, w, h)
        
        Returns:
            损失值
        """
        if not cross_attn_weights:
            return 0.0
        
        # 计算正确边界框对应的补丁索引
        patch_indices = calculate_bbox_patch_indices(bbox, self.patch_size, self.search_size)
        
        if not patch_indices:
            return 0.0
        
        total_loss = 0.0
        for coord_attn in cross_attn_weights:
            for layer_attn in coord_attn:
                # 提取对搜索区域特征的注意力权重
                search_seq_len = (self.search_size // self.patch_size) ** 2
                search_attn = layer_attn[:, :, :, :search_seq_len]
                
                # 计算正确补丁的平均注意力权重
                correct_attn = search_attn[:, :, :, patch_indices].mean(dim=-1)
                
                # 损失：降低这些权重
                loss = correct_attn.mean()
                total_loss += loss
        
        return total_loss / (len(cross_attn_weights) * len(cross_attn_weights[0]))


class AttentionLoss(nn.Module):
    """
    组合注意力损失函数
    """
    def __init__(self, patch_size=16, search_size=384, encoder_weight=1.0, decoder_weight=1.0):
        super().__init__()
        self.encoder_loss = EncoderAttentionLoss(patch_size, search_size)
        self.decoder_loss = DecoderAttentionLoss(patch_size, search_size)
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight
    
    def forward(self, encoder_attn_weights, cross_attn_weights, bbox):
        """
        计算组合注意力损失
        
        Args:
            encoder_attn_weights: 编码器注意力权重
            cross_attn_weights: 解码器交叉注意力权重
            bbox: 正确的边界框坐标
        
        Returns:
            组合损失值
        """
        encoder_loss = self.encoder_loss(encoder_attn_weights, bbox)
        decoder_loss = self.decoder_loss(cross_attn_weights, bbox)
        
        total_loss = self.encoder_weight * encoder_loss + self.decoder_weight * decoder_loss
        return total_loss


# 辅助函数：计算注意力权重的统计信息
def get_attention_stats(attn_weights):
    """
    计算注意力权重的统计信息
    
    Args:
        attn_weights: 注意力权重列表
    
    Returns:
        统计信息字典
    """
    if not attn_weights:
        return {}
    
    stats = {}
    for i, layer_attn in enumerate(attn_weights):
        layer_stats = {
            'mean': float(layer_attn.mean().item()),
            'max': float(layer_attn.max().item()),
            'min': float(layer_attn.min().item()),
            'std': float(layer_attn.std().item())
        }
        stats[f'layer_{i}'] = layer_stats
    
    return stats


# 辅助函数：可视化注意力权重
def visualize_attention(attn_weights, patch_size, image_size, save_path=None):
    """
    可视化注意力权重
    
    Args:
        attn_weights: 注意力权重，形状为 [batch, num_heads, seq_len, seq_len]
        patch_size: 补丁大小
        image_size: 图像大小
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 取第一个样本，第一个头的注意力权重
    attn = attn_weights[0, 0].detach().cpu().numpy()
    
    # 计算补丁数量
    num_patches = image_size // patch_size
    
    # 取搜索区域内部的注意力
    search_seq_len = num_patches * num_patches
    attn = attn[:search_seq_len, :search_seq_len]
    
    # 可视化注意力热图
    plt.figure(figsize=(10, 10))
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Attention Weights Heatmap')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
