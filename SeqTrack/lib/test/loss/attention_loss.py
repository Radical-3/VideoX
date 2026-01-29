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
    搜索区域注意力损失函数
    """
    def __init__(self, patch_size=16, search_size=384):
        super().__init__()
        self.decoder_loss = DecoderAttentionLoss(patch_size, search_size)
    
    def forward(self, attention_weights, bbox, crop_info=None):
        """
        计算搜索区域注意力损失
        
        Args:
            attention_weights: 注意力权重字典，包含编码器和解码器的注意力权重
            bbox: 正确的边界框坐标
            crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        
        Returns:
            损失值
        """
        # 只使用解码器的注意力权重（搜索区域相关）
        cross_attn_weights = []
        
        for layer_name, weights in attention_weights.items():
            if 'decoder' in layer_name:
                cross_attn_weights.append(weights)
        
        # 只返回解码器损失，不计算编码器损失
        decoder_loss = self.decoder_loss(cross_attn_weights, bbox, crop_info)
        return decoder_loss


def calculate_foreground_background_scores(score_map, bbox, crop_info=None, patch_size=16, search_size=256):
    """
    计算前景和背景区域的得分
    
    Args:
        score_map: 得分图，形状为 [batch, channels, height, width] 或 [batch, channels, features] 或 NumPy 数组
        bbox: 边界框坐标 (x, y, w, h)
        crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        patch_size: 补丁大小
        search_size: 搜索区域大小
    
    Returns:
        foreground_max: 前景区域的最大得分
        background_max: 背景区域的最大得分
    """
    # 调试：打印输入信息
    print(f"\n=== 调试得分损失计算 ===")
    print(f"原始边界框: {bbox}")
    print(f"裁剪信息: {crop_info}")
    print(f"得分图形状: {score_map.shape}, 类型: {type(score_map)}")
    
    # 检查得分图类型
    is_numpy = isinstance(score_map, np.ndarray)
    print(f"得分图类型: {'NumPy数组' if is_numpy else 'PyTorch张量'}")
    
    # 处理不同格式的得分图
    print(f"得分图形状: {score_map.shape}, 维度数: {len(score_map.shape)}")
    
    # 首先检查是否是特征向量格式
    # 特征向量格式通常是 [batch, channels, features] 或 [channels, features]
    if len(score_map.shape) == 3:
        print(f"3维得分图 - 各维度大小: {score_map.shape}")
        # 检查最后一维是否大于1000
        if score_map.shape[2] > 1000:
            print("检测到特征向量格式的得分图")
            # 对于特征向量格式，直接计算最大值
            if is_numpy:
                foreground_max = float(np.max(score_map))
                background_max = float(np.min(score_map))  # 使用最小值作为背景得分
            else:
                foreground_max = score_map.max().item()
                background_max = score_map.min().item()
            print(f"特征向量格式 - 前景最大: {foreground_max}, 背景最大: {background_max}")
            return foreground_max, background_max
        else:
            print(f"最后一维大小为 {score_map.shape[2]}，不认为是特征向量格式")
            # 添加batch维度
            if is_numpy:
                score_map = np.expand_dims(score_map, axis=0)
            else:
                score_map = score_map.unsqueeze(0)
            print(f"添加batch维度后的形状: {score_map.shape}")
    # 特殊处理：如果是特征向量格式但维度不是3
    elif len(score_map.shape) == 2:
        print("检测到2D特征向量格式的得分图")
        if is_numpy:
            foreground_max = float(np.max(score_map))
            background_max = float(np.min(score_map))
        else:
            foreground_max = score_map.max().item()
            background_max = score_map.min().item()
        print(f"2D特征向量格式 - 前景最大: {foreground_max}, 背景最大: {background_max}")
        return foreground_max, background_max
    elif len(score_map.shape) != 4:
        # 如果维度不是4维，返回0
        print(f"得分图维度不是4维，返回0")
        return 0.0, 0.0
    
    # 以下是原始的空间得分图处理逻辑...
    # 处理坐标转换
    if crop_info:
        # 将原始图像坐标系的边界框转换为裁剪图像坐标系
        crop_bbox = convert_bbox_to_crop_coords(
            bbox, 
            crop_info['crop_center'], 
            crop_info['crop_size'], 
            crop_info['search_size']
        )
        print(f"转换后的裁剪边界框: {crop_bbox}")
    else:
        # 如果没有裁剪信息，假设边界框已经是相对于裁剪图像的
        crop_bbox = bbox
        print(f"使用原始边界框作为裁剪边界框: {crop_bbox}")
    
    # 计算前景区域的坐标
    x, y, w, h = crop_bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # 确保坐标在有效范围内
    h_map, w_map = score_map.shape[2], score_map.shape[3]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_map, x2)
    y2 = min(h_map, y2)
    
    print(f"前景区域坐标: ({x1}, {y1}) - ({x2}, {y2})")
    print(f"得分图尺寸: 高={h_map}, 宽={w_map}")
    
    # 检查前景区域是否有效
    if x1 >= x2 or y1 >= y2:
        print("前景区域无效，返回0")
        return 0.0, 0.0
    
    # 提取前景区域的得分
    foreground_score = score_map[:, :, y1:y2, x1:x2]
    print(f"前景区域得分形状: {foreground_score.shape}")
    
    if foreground_score.size == 0:
        print("前景区域得分大小为0，返回0")
        return 0.0, 0.0
    
    # 计算前景区域的最大得分
    if is_numpy:
        foreground_max = float(np.max(foreground_score))
    else:
        foreground_max = foreground_score.max().item()
    
    print(f"前景区域最大得分: {foreground_max}")
    
    # 计算背景区域的得分
    # 创建一个掩码，前景区域为0，背景区域为1
    if is_numpy:
        mask = np.ones_like(score_map)
        mask[:, :, y1:y2, x1:x2] = 0
    else:
        mask = torch.ones_like(score_map)
        mask[:, :, y1:y2, x1:x2] = 0
    
    # 提取背景区域的得分
    background_score = score_map * mask
    print(f"背景区域得分形状: {background_score.shape}")
    
    if background_score.size == 0:
        print("背景区域得分大小为0，返回前景得分")
        return foreground_max, 0.0
    
    # 计算背景区域的最大得分
    if is_numpy:
        background_max = float(np.max(background_score))
    else:
        background_max = background_score.max().item()
    
    print(f"背景区域最大得分: {background_max}")
    
    return foreground_max, background_max


class ScoreLoss(nn.Module):
    def __init__(self, patch_size=16, search_size=256):
        super(ScoreLoss, self).__init__()
        self.patch_size = patch_size
        self.search_size = search_size
    
    def forward(self, score_map, bbox, crop_info=None):
        """
        计算得分损失
        
        Args:
            score_map: 得分图
            bbox: 边界框 (x, y, w, h)
            crop_info: 裁剪信息
        
        Returns:
            loss: 得分损失
        """
        if score_map is None:
            print("得分图为None，返回0")
            return 0.0
        
        # 计算前景和背景区域的最大得分
        foreground_max, background_max = calculate_foreground_background_scores(
            score_map, bbox, crop_info, self.patch_size, self.search_size
        )
        
        # 计算得分损失：前景最大得分 - 背景最大得分
        # 目标是让这个值尽可能小，即前景得分低，背景得分高
        loss = foreground_max - background_max
        print(f"计算得到的得分损失: {loss}")
        
        return loss


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
