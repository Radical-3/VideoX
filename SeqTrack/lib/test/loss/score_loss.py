import torch
import torch.nn as nn
import numpy as np


def calculate_foreground_background_scores(score_map, bbox, crop_info=None, patch_size=16, search_size=256):
    """
    计算前景和背景区域的得分
    
    Args:
        score_map: 得分图，形状为 [batch, 4, bins] 对应SeqTrack的词空间格式
        bbox: 边界框坐标 (x, y, w, h)
        crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        patch_size: 补丁大小
        search_size: 搜索区域大小
    
    Returns:
        foreground_max: 正确边界框对应词空间的得分
        background_max: 其他词空间的最大得分
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
    
    # 检查是否是SeqTrack的词空间格式 [batch, 4, bins]
    if len(score_map.shape) == 3 and score_map.shape[1] == 4:
        print("检测到SeqTrack词空间格式的得分图")
        
        # 确保边界框是相对于搜索区域的
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
        
        # 计算每个坐标维度的前景得分和背景得分
        total_foreground = 0.0
        total_background = 0.0
        
        # SeqTrack的词空间范围是0-1，映射到bins个bin
        bins = score_map.shape[2]
        print(f"词空间bin数量: {bins}")
        
        # SeqTrack使用中心坐标格式 (cx, cy, w, h)，而不是左上角坐标格式 (x, y, w, h)
        # 严格按照SeqTrack的训练代码实现映射逻辑
        x, y, w, h = crop_bbox
        
        # 转换为x0y0x1y1格式
        x1 = x + w
        y1 = y + h
        
        # 归一化到 [0, 1] 范围
        x_normalized = x / search_size
        y_normalized = y / search_size
        x1_normalized = x1 / search_size
        y1_normalized = y1 / search_size
        
        # 确保在有效范围内
        x_normalized = max(0.0, min(1.0, x_normalized))
        y_normalized = max(0.0, min(1.0, y_normalized))
        x1_normalized = max(0.0, min(1.0, x1_normalized))
        y1_normalized = max(0.0, min(1.0, y1_normalized))
        
        # 转换为cxcywh格式
        cx_normalized = (x_normalized + x1_normalized) / 2
        cy_normalized = (y_normalized + y1_normalized) / 2
        w_normalized = x1_normalized - x_normalized
        h_normalized = y1_normalized - y_normalized
        
        # 计算对应的bin索引，严格按照SeqTrack的训练代码：乘以(bins-1)，然后取整
        cx_bin = int(cx_normalized * (bins - 1))
        cy_bin = int(cy_normalized * (bins - 1))
        w_bin = int(w_normalized * (bins - 1))
        h_bin = int(h_normalized * (bins - 1))
        
        # 确保bin索引在有效范围内
        cx_bin = max(0, min(bins - 1, cx_bin))
        cy_bin = max(0, min(bins - 1, cy_bin))
        w_bin = max(0, min(bins - 1, w_bin))
        h_bin = max(0, min(bins - 1, h_bin))
        
        print(f"中心坐标 - cx: {x + w/2}, cy: {y + h/2}, w: {w}, h: {h}")
        print(f"归一化中心坐标 - cx: {cx_normalized}, cy: {cy_normalized}, w: {w_normalized}, h: {h_normalized}")
        print(f"对应的bin索引 - cx: {cx_bin}, cy: {cy_bin}, w: {w_bin}, h: {h_bin}")
        
        # 遍历4个坐标维度
        bin_indices = [cx_bin, cy_bin, w_bin, h_bin]
        for i in range(4):  # 4个坐标维度: cx, cy, w, h
            # 提取当前维度的得分图
            if is_numpy:
                dim_scores = score_map[0, i, :]  # 假设batch_size=1
                
                # 找到得分最高的bin索引（模型预测的bin）
                pred_bin_idx = np.argmax(dim_scores)
                # 预测的坐标值
                pred_value = (pred_bin_idx / (bins - 1)) * search_size
                
                # 真实边界框对应bin的索引
                true_bin_idx = bin_indices[i]
                # 前景得分：真实边界框对应bin的得分
                foreground_score = dim_scores[true_bin_idx]
                # 背景得分：其他所有bin的得分的最大值
                # 创建掩码，排除当前bin
                mask = np.ones_like(dim_scores)
                mask[true_bin_idx] = 0
                background_scores = dim_scores * mask
                background_score = np.max(background_scores) if np.any(mask) else 0.0
            else:
                dim_scores = score_map[0, i, :]  # 假设batch_size=1
                
                # 找到得分最高的bin索引（模型预测的bin）
                pred_bin_idx = dim_scores.argmax().item()
                # 预测的坐标值
                pred_value = (pred_bin_idx / (bins - 1)) * search_size
                
                # 真实边界框对应bin的索引
                true_bin_idx = bin_indices[i]
                # 前景得分：真实边界框对应bin的得分
                foreground_score = dim_scores[true_bin_idx].item()
                # 背景得分：其他所有bin的得分的最大值
                # 创建掩码，排除当前bin
                mask = torch.ones_like(dim_scores)
                mask[true_bin_idx] = 0
                background_scores = dim_scores * mask
                background_score = background_scores.max().item() if mask.sum() > 0 else 0.0
            
            print(f"维度 {i} - 预测值: {pred_value}")
            print(f"维度 {i} - 真实bin索引: {true_bin_idx}, 预测bin索引: {pred_bin_idx}")
            print(f"维度 {i} - 真实bin得分: {foreground_score}, 其他bin最高得分: {background_score}")
            
            total_foreground += foreground_score
            total_background += background_score
        
        # 计算平均前景得分和背景得分
        foreground_max = total_foreground / 4.0
        background_max = total_background / 4.0
        
        # 打印得分图的统计信息，帮助理解得分图的含义
        if is_numpy:
            all_scores = score_map.flatten()
            print(f"得分图统计信息 - 最小值: {np.min(all_scores)}, 最大值: {np.max(all_scores)}, 平均值: {np.mean(all_scores)}")
        else:
            all_scores = score_map.view(-1)
            print(f"得分图统计信息 - 最小值: {all_scores.min().item()}, 最大值: {all_scores.max().item()}, 平均值: {all_scores.mean().item()}")
        
        print(f"SeqTrack词空间格式 - 真实边界框对应得分: {foreground_max}, 其他词空间最高得分: {background_max}")
        return foreground_max, background_max
    
    # 处理其他格式的得分图
    elif len(score_map.shape) == 3:
        print(f"3维得分图 - 各维度大小: {score_map.shape}")
        # 检查最后一维是否大于1000
        if score_map.shape[2] > 1000:
            print("检测到特征向量格式的得分图")
            # 对于特征向量格式，将最大值作为前景得分，第二大值作为背景得分
            if is_numpy:
                # 扁平化得分图
                flattened_scores = score_map.flatten()
                # 计算最大值和第二大值
                if len(flattened_scores) > 1:
                    sorted_scores = np.sort(flattened_scores)[::-1]
                    foreground_max = float(sorted_scores[0])  # 当前最高得分
                    background_max = float(sorted_scores[1])  # 其他区域的最高得分
                else:
                    foreground_max = float(np.max(score_map))
                    background_max = 0.0
            else:
                # 扁平化得分图
                flattened_scores = score_map.view(-1)
                # 计算最大值和第二大值
                if flattened_scores.numel() > 1:
                    sorted_scores, _ = torch.sort(flattened_scores, descending=True)
                    foreground_max = sorted_scores[0].item()  # 当前最高得分
                    background_max = sorted_scores[1].item()  # 其他区域的最高得分
                else:
                    foreground_max = score_map.max().item()
                    background_max = 0.0
            print(f"特征向量格式 - 最高得分: {foreground_max}, 第二高得分: {background_max}")
            return foreground_max, background_max
        else:
            print(f"最后一维大小为 {score_map.shape[2]}，不认为是特征向量格式")
            return 0.0, 0.0
    elif len(score_map.shape) == 2:
        print("检测到2D特征向量格式的得分图")
        if is_numpy:
            # 扁平化得分图
            flattened_scores = score_map.flatten()
            # 计算最大值和第二大值
            if len(flattened_scores) > 1:
                sorted_scores = np.sort(flattened_scores)[::-1]
                foreground_max = float(sorted_scores[0])  # 当前最高得分
                background_max = float(sorted_scores[1])  # 其他区域的最高得分
            else:
                foreground_max = float(np.max(score_map))
                background_max = 0.0
        else:
            # 扁平化得分图
            flattened_scores = score_map.view(-1)
            # 计算最大值和第二大值
            if flattened_scores.numel() > 1:
                sorted_scores, _ = torch.sort(flattened_scores, descending=True)
                foreground_max = sorted_scores[0].item()  # 当前最高得分
                background_max = sorted_scores[1].item()  # 其他区域的最高得分
            else:
                foreground_max = score_map.max().item()
                background_max = 0.0
        print(f"2D特征向量格式 - 最高得分: {foreground_max}, 第二高得分: {background_max}")
        return foreground_max, background_max
    else:
        # 其他格式返回0
        print(f"得分图格式不支持，返回0")
        return 0.0, 0.0



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
        # 前景区域：正确边界框对应的区域
        # 背景区域：其他所有区域
        foreground_max, background_max = calculate_foreground_background_scores(
            score_map, bbox, crop_info, self.patch_size, self.search_size
        )
        
        # 计算得分损失：前景最大得分 - 背景最大得分
        # 目标是让这个值尽可能小，实现：
        # 1. 前景得分低（正确边界框对应的得分降低）
        # 2. 背景得分高（其他区域的得分最大值提高）
        loss = foreground_max - background_max
        print(f"计算得到的得分损失: {loss}")
        
        return loss
