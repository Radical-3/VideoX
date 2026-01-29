import torch
import torch.nn as nn
import numpy as np


def convert_bbox_to_crop_coords(bbox, crop_center, crop_size, image_size):
    """
    将原始图像坐标系的边界框转换为裁剪图像坐标系的边界框
    
    Args:
        bbox: 原始图像坐标系的边界框 (x, y, w, h)
        crop_center: 裁剪区域的中心坐标 (cx, cy)（相对于原始图像）
        crop_size: 裁剪区域的大小（在调整大小之前）
        image_size: 调整大小后的裁剪图像大小
    
    Returns:
        裁剪图像坐标系的边界框 (x, y, w, h)
    """
    x, y, w, h = bbox
    cx, cy = crop_center
    
    # 计算裁剪区域的左上角坐标（相对于原始图像）
    crop_x1 = cx - crop_size / 2
    crop_y1 = cy - crop_size / 2
    
    # 计算边界框在裁剪区域中的坐标
    crop_x = (x - crop_x1) * (image_size / crop_size)
    crop_y = (y - crop_y1) * (image_size / crop_size)
    crop_w = w * (image_size / crop_size)
    crop_h = h * (image_size / crop_size)
    
    # 确保边界框在裁剪图像内
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    crop_w = min(image_size - crop_x, crop_w)
    crop_h = min(image_size - crop_y, crop_h)
    
    return [crop_x, crop_y, crop_w, crop_h]


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
    
    def forward(self, attn_weights, bbox, crop_info=None):
        """
        计算编码器注意力损失
        
        Args:
            attn_weights: 编码器注意力权重列表，每个元素形状为 [batch, num_heads, seq_len, seq_len]
            bbox: 正确的边界框坐标 (x, y, w, h)
            crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        
        Returns:
            损失值
        """
        # 检查注意力权重是否存在
        if not attn_weights:
            return 0.0
        
        # 处理坐标转换
        if crop_info:
            # 将原始图像坐标系的边界框转换为裁剪图像坐标系
            crop_bbox = convert_bbox_to_crop_coords(
                bbox, 
                crop_info['crop_center'], 
                crop_info['crop_size'], 
                crop_info['search_size']
            )
        else:
            # 如果没有裁剪信息，假设边界框已经是相对于裁剪图像的
            crop_bbox = bbox
        
        # 计算正确边界框对应的补丁索引
        patch_indices = calculate_bbox_patch_indices(crop_bbox, self.patch_size, self.search_size)
        
        # 检查补丁索引是否为空
        if not patch_indices:
            return 0.0
        
        # 只使用中间层的注意力权重
        num_layers = len(attn_weights)
        if num_layers == 0:
            return 0.0
        
        # 选择中间层
        if num_layers % 2 == 1:
            # 奇数层，选择中间那一层
            middle_idx = num_layers // 2
            target_layers = [attn_weights[middle_idx]]
        else:
            # 偶数层，选择中间两层
            middle_idx1 = num_layers // 2 - 1
            middle_idx2 = num_layers // 2
            target_layers = [attn_weights[middle_idx1], attn_weights[middle_idx2]]
        
        # 检查目标层是否为空
        if not target_layers:
            return 0.0
        
        total_loss = 0.0
        for layer_attn in target_layers:
            # 计算搜索区域的序列长度
            search_seq_len = (self.search_size // self.patch_size) ** 2
            
            # 检查layer_attn的维度
            if len(layer_attn.shape) == 3:
                # 如果是3维数组，添加batch维度
                layer_attn = np.expand_dims(layer_attn, axis=0)
            elif len(layer_attn.shape) != 4:
                # 如果维度不是4维，跳过
                continue
            
            # 确保序列长度正确
            if layer_attn.shape[-1] < search_seq_len:
                continue
            
            # 提取搜索区域内部的注意力权重
            # 注意：这里我们使用整个序列，而不是只使用搜索区域
            # 这样可以确保我们捕获所有相关的注意力
            search_attn = layer_attn
            
            # 计算正确补丁的平均注意力权重
            # 确保patch_indices在有效范围内
            max_idx = search_attn.shape[-1]
            valid_indices = [idx for idx in patch_indices if idx < max_idx]
            
            if not valid_indices:
                continue
            
            # 处理NumPy数组和PyTorch张量的情况
            if isinstance(search_attn, np.ndarray):
                # NumPy数组使用axis参数
                correct_attn = search_attn[:, :, :, valid_indices].mean(axis=-1)
            else:
                # PyTorch张量使用dim参数
                correct_attn = search_attn[:, :, :, valid_indices].mean(dim=-1)
            
            # 损失：降低这些权重
            loss = correct_attn.mean()
            total_loss += loss
        
        return total_loss / len(target_layers) if target_layers else 0.0


class DecoderAttentionLoss(nn.Module):
    """
    解码器注意力损失函数
    目标：降低解码器对正确边界框对应特征的注意力权重
    """
    def __init__(self, patch_size=16, search_size=384):
        super().__init__()
        self.patch_size = patch_size
        self.search_size = search_size
    
    def forward(self, cross_attn_weights, bbox, crop_info=None):
        """
        计算解码器交叉注意力损失
        
        Args:
            cross_attn_weights: 解码器交叉注意力权重列表，每个元素是一个列表，包含每一层的注意力权重
            bbox: 正确的边界框坐标 (x, y, w, h)
            crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        
        Returns:
            损失值
        """
        # 检查注意力权重是否存在
        if not cross_attn_weights:
            return 0.0
        
        # 处理坐标转换
        if crop_info:
            # 将原始图像坐标系的边界框转换为裁剪图像坐标系
            crop_bbox = convert_bbox_to_crop_coords(
                bbox, 
                crop_info['crop_center'], 
                crop_info['crop_size'], 
                crop_info['search_size']
            )
        else:
            # 如果没有裁剪信息，假设边界框已经是相对于裁剪图像的
            crop_bbox = bbox
        
        # 计算正确边界框对应的补丁索引
        patch_indices = calculate_bbox_patch_indices(crop_bbox, self.patch_size, self.search_size)
        
        # 检查补丁索引是否为空
        if not patch_indices:
            return 0.0
        
        # 只使用中间层的注意力权重
        total_loss = 0.0
        num_layers_used = 0
        
        for coord_attn in cross_attn_weights:
            num_layers = len(coord_attn)
            if num_layers == 0:
                continue
            
            # 选择中间层
            if num_layers % 2 == 1:
                # 奇数层，选择中间那一层
                middle_idx = num_layers // 2
                target_layers = [coord_attn[middle_idx]]
            else:
                # 偶数层，选择中间两层
                middle_idx1 = num_layers // 2 - 1
                middle_idx2 = num_layers // 2
                target_layers = [coord_attn[middle_idx1], coord_attn[middle_idx2]]
            
            for layer_attn in target_layers:
                # 计算搜索区域的序列长度
                search_seq_len = (self.search_size // self.patch_size) ** 2
                
                # 检查layer_attn的维度
                if len(layer_attn.shape) == 3:
                    # 如果是3维数组，添加batch维度
                    layer_attn = np.expand_dims(layer_attn, axis=0)
                elif len(layer_attn.shape) != 4:
                    # 如果维度不是4维，跳过
                    continue
                
                # 确保序列长度正确
                if layer_attn.shape[-1] < search_seq_len:
                    continue
                
                # 提取对搜索区域特征的注意力权重
                search_attn = layer_attn[:, :, :, :search_seq_len]
                
                # 确保patch_indices在有效范围内
                max_idx = search_attn.shape[-1]
                valid_indices = [idx for idx in patch_indices if idx < max_idx]
                
                if not valid_indices:
                    continue
                
                # 计算正确补丁的平均注意力权重
                # 处理NumPy数组和PyTorch张量的情况
                if isinstance(search_attn, np.ndarray):
                    # NumPy数组使用axis参数
                    correct_attn = search_attn[:, :, :, valid_indices].mean(axis=-1)
                else:
                    # PyTorch张量使用dim参数
                    correct_attn = search_attn[:, :, :, valid_indices].mean(dim=-1)
                
                # 损失：降低这些权重
                loss = correct_attn.mean()
                total_loss += loss
                num_layers_used += 1
        
        return total_loss / num_layers_used if num_layers_used > 0 else 0.0


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
    
    def forward(self, encoder_attn_weights, cross_attn_weights, bbox, crop_info=None):
        """
        计算组合注意力损失
        
        Args:
            encoder_attn_weights: 编码器注意力权重
            cross_attn_weights: 解码器交叉注意力权重
            bbox: 正确的边界框坐标
            crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        
        Returns:
            损失值
        """
        encoder_loss = self.encoder_loss(encoder_attn_weights, bbox, crop_info)
        decoder_loss = self.decoder_loss(cross_attn_weights, bbox, crop_info)
        return self.encoder_weight * encoder_loss + self.decoder_weight * decoder_loss


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
