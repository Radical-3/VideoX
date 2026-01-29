import numpy as np
import torch
import cv2
import os

# 添加项目根目录到Python路径
import _init_paths

from lib.test.evaluation import get_dataset
from lib.test.tracker.seqtrack import SEQTRACK
from lib.test.parameter.seqtrack import parameters
from lib.test.loss.attention_loss import AttentionLoss
from lib.test.loss.score_loss import ScoreLoss


class AdversarialAttack:
    def __init__(self, dataset_name='custom_test'):
        self.dataset_name = dataset_name
        # 获取参数
        # 对于参数，我们仍然使用'custom'，因为这是配置文件的名称
        self.params = parameters('custom')
        # 创建跟踪器
        # 对于跟踪器，我们使用完整的数据集名称
        self.tracker = SEQTRACK(self.params, dataset_name)
        # 启用注意力权重输出
        self.tracker.enable_attention_output()
        # 创建注意力损失函数
        self.attention_loss = AttentionLoss(patch_size=16, search_size=256)
        # 创建得分损失函数
        self.score_loss = ScoreLoss(patch_size=16, search_size=256)
    
    def calculate_iou_loss(self, pred_bbox, gt_bbox):
        """计算IoU损失
        
        args:
            pred_bbox: 预测边界框 [x, y, w, h]
            gt_bbox: 真实边界框 [x, y, w, h]
        
        returns:
            float: IoU损失值 (1 - IoU)
        """
        # 确保边界框格式正确
        def ensure_bbox_format(bbox):
            if len(bbox) == 4:
                # 检查是否是[x1, y1, x2, y2]格式
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # 转换为[x, y, w, h]格式
                    x = bbox[0]
                    y = bbox[1]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    return [x, y, w, h]
            return bbox
        
        pred_bbox = ensure_bbox_format(pred_bbox)
        gt_bbox = ensure_bbox_format(gt_bbox)
        
        # 计算IoU
        x1 = max(pred_bbox[0], gt_bbox[0])
        y1 = max(pred_bbox[1], gt_bbox[1])
        x2 = min(pred_bbox[0] + pred_bbox[2], gt_bbox[0] + gt_bbox[2])
        y2 = min(pred_bbox[1] + pred_bbox[3], gt_bbox[1] + gt_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = pred_bbox[2] * pred_bbox[3] + gt_bbox[2] * gt_bbox[3] - intersection
        
        if union == 0:
            return 1.0
        
        iou = intersection / union
        return iou
    
    def calculate_attention_loss(self, encoder_attn_weights, cross_attn_weights, gt_bbox, crop_info=None):
        """计算注意力损失
        
        args:
            encoder_attn_weights: 编码器注意力权重
            cross_attn_weights: 解码器交叉注意力权重
            gt_bbox: 真实边界框 [x, y, w, h]
            crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        
        returns:
            float: 注意力损失值
        """
        if encoder_attn_weights is None or cross_attn_weights is None:
            return 0.0
        
        # 计算注意力损失
        from lib.test.loss.attention_loss import AttentionLoss
        attention_loss = AttentionLoss(patch_size=16, search_size=256)
        # 合并编码器和解码器的注意力权重
        attention_weights = {}
        if encoder_attn_weights:
            # 将编码器注意力权重列表转换为字典
            for i, weights in enumerate(encoder_attn_weights):
                attention_weights[f'encoder_layer_{i}'] = weights
        if cross_attn_weights:
            # 将解码器注意力权重列表转换为字典
            for i, layer_weights in enumerate(cross_attn_weights):
                attention_weights[f'decoder_layer_{i}'] = layer_weights
        loss = attention_loss(
            attention_weights=attention_weights,
            bbox=gt_bbox,
            crop_info=crop_info
        )
        
        return loss.item()
    
    def calculate_score_loss(self, score_map, gt_bbox, crop_info=None):
        """计算得分损失
        
        args:
            score_map: 得分图，形状为 [batch, channels, height, width]
            gt_bbox: 真实边界框 [x, y, w, h]
            crop_info: 裁剪信息字典，包含 'crop_center' (cx, cy), 'crop_size' (裁剪大小), 'search_size' (调整后的大小)
        
        returns:
            float: 得分损失值
        """
        if score_map is None:
            print("得分图为None，返回0")
            return 0.0
        
        # 调试：打印得分图的基本信息
        print(f"得分图形状: {score_map.shape}, 类型: {type(score_map)}")
        print(f"得分图最小值: {score_map.min()}, 最大值: {score_map.max()}, 平均值: {score_map.mean()}")
        
        # 计算得分损失
        loss = self.score_loss(
            score_map=score_map,
            bbox=gt_bbox,
            crop_info=crop_info
        )
        
        print(f"计算得到的得分损失: {loss}")
        return loss
    
    def load_frame(self, frame_path):
        """加载帧数据
        
        args:
            frame_path: 帧文件路径
        
        returns:
            numpy.ndarray: 图像数据
        """
        try:
            # 加载npy文件
            data = np.load(frame_path, allow_pickle=True)
            
            # 处理不同格式的数据
            if isinstance(data, dict):
                image = data['image']
            elif data.ndim == 0:
                data = data.item()
                if isinstance(data, dict) and 'image' in data:
                    image = data['image']
                else:
                    print(f"无法加载帧图像: {frame_path}")
                    return None
            elif data.shape == (1,):
                data = data[0]
                if isinstance(data, dict) and 'image' in data:
                    image = data['image']
                else:
                    print(f"无法加载帧图像: {frame_path}")
                    return None
            elif data.shape == (6,):
                # 形状为(6,)的数组，顺序: identifier, image, image_mask, label, camera_position, relative_remove
                image = data[1]
            else:
                print(f"无法加载帧图像: {frame_path}")
                return None
            
            return image
        except Exception as e:
            print(f"加载帧时出错: {e}")
            return None
    
    def attack(self):
        """执行对抗攻击
        """
        # 获取数据集
        dataset = get_dataset(self.dataset_name)
        print(f"\n=== 加载数据集完成，共 {len(dataset)} 个序列 ===")
        
        # 遍历每个序列
        for seq_idx, seq in enumerate(dataset):
            print(f"\n=== 处理序列 {seq_idx + 1}/{len(dataset)}: {seq.name} ===")
            print(f"总帧数: {len(seq.frames)}")
            
            # 第一帧作为模板
            first_frame = self.load_frame(seq.frames[0])
            if first_frame is None:
                print(f"无法加载第一帧: {seq.frames[0]}")
                continue
            
            # 初始化跟踪器
            init_bbox = seq.init_bbox()
            print(f"初始边界框: {init_bbox}")
            
            self.tracker.initialize(first_frame, {'init_bbox': init_bbox})
            
            # 处理后续帧
            for frame_idx in range(1, len(seq.frames)):
                # 加载当前帧
                frame_path = seq.frames[frame_idx]
                frame = self.load_frame(frame_path)
                if frame is None:
                    continue
                
                # 获取真实边界框
                gt_bbox = seq.ground_truth_rect[frame_idx]
                
                # 跟踪预测
                output = self.tracker.track(frame)
                
                # 提取预测结果
                pred_bbox = output['target_bbox']
                encoder_attn_weights = output.get('encoder_attn_weights', None)
                cross_attn_weights = output.get('cross_attn_weights', None)
                
                # 计算裁剪信息
                # 裁剪中心是上一帧预测边界框的中心
                prev_bbox = self.tracker.state
                prev_cx = prev_bbox[0] + prev_bbox[2] / 2
                prev_cy = prev_bbox[1] + prev_bbox[3] / 2
                
                # 裁剪大小是基于预测边界框的大小和搜索因子
                crop_size = (prev_bbox[2] * prev_bbox[3]) ** 0.5 * self.params.search_factor
                
                # 调整后的搜索图像大小
                search_size = self.params.search_size
                
                crop_info = {
                    'crop_center': (prev_cx, prev_cy),
                    'crop_size': crop_size,
                    'search_size': search_size
                }
                
                # 提取得分图
                score_map = output.get('score_maps', None)
                
                # 调试：打印得分图信息
                if score_map is not None:
                    print(f"得分图形状: {score_map.shape}, 类型: {type(score_map)}")
                else:
                    print("得分图为None")
                
                # 计算损失函数
                iou_loss = self.calculate_iou_loss(pred_bbox, gt_bbox)
                attention_loss = self.calculate_attention_loss(encoder_attn_weights, cross_attn_weights, gt_bbox, crop_info)
                score_loss = self.calculate_score_loss(score_map, gt_bbox, crop_info)
                
                # 打印结果
                print(f"\n=== 帧 {frame_idx} ===")
                print(f"预测边界框: {pred_bbox}")
                print(f"真实边界框: {gt_bbox}")
                print(f"IoU损失: {iou_loss:.4f}")
                print(f"注意力损失: {attention_loss:.4f}")
                print(f"得分损失: {score_loss:.4f}")
                print(f"裁剪中心: {crop_info['crop_center']}")
                print(f"裁剪大小: {crop_info['crop_size']:.2f}")
                
                # 每处理5帧打印一次进度
                if (frame_idx + 1) % 5 == 0:
                    print(f"\n进度: {frame_idx + 1}/{len(seq.frames)}")
            
            # 打印序列完成信息
            print(f"\n=== 序列 {seq.name} 处理完成 ===")
        
        # 打印所有序列处理完成信息
        print(f"\n=== 所有 {len(dataset)} 个序列处理完成 ===")


if __name__ == '__main__':
    # 创建对抗攻击实例
    attacker = AdversarialAttack(dataset_name='custom_test')
    # 执行攻击
    attacker.attack()
