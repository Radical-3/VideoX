import torch
import torch.nn as nn
import numpy as np


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


def convert_bbox_to_crop_coords(bbox, crop_center, crop_size, search_size):
    """
    将原始图像坐标系中的边界框转换为裁剪图像坐标系
    
    Args:
        bbox: 原始图像中的边界框 (x, y, w, h)
        crop_center: 裁剪中心 (cx, cy)
        crop_size: 裁剪大小
        search_size: 调整后的搜索区域大小
    
    Returns:
        crop_bbox: 裁剪图像坐标系中的边界框 (x, y, w, h)
    """
    x, y, w, h = bbox
    cx, cy = crop_center
    
    # 计算裁剪图像中的坐标
    # 首先将原始坐标转换为相对于裁剪中心的坐标
    crop_x = (x - (cx - crop_size / 2)) * (search_size / crop_size)
    crop_y = (y - (cy - crop_size / 2)) * (search_size / crop_size)
    crop_w = w * (search_size / crop_size)
    crop_h = h * (search_size / crop_size)
    
    # 确保坐标为正数
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    
    return [crop_x, crop_y, crop_w, crop_h]


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
