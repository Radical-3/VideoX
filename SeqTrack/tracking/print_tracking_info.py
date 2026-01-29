import os
import sys
import numpy as np
import cv2

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lib.test.evaluation import get_dataset
from lib.test.tracker.seqtrack import SEQTRACK
from lib.test.utils.load_text import load_text
from lib.test.parameter.seqtrack import parameters

# 参数定义
class Parameters:
    def __init__(self):
        # 数据集设置
        self.dataset_name = 'custom_test'  # 自定义数据集测试集
        self.dataset_path = r'd:\code\VideoX\SeqTrack\data\custom'  # 数据集路径
        
        # 跟踪器设置
        self.tracker_name = 'seqtrack'
        self.tracker_param = 'seqtrack_b256_got'
        
        # 测试设置
        self.threads = 0  # 单线程运行
        self.visualize = False  # 是否可视化
        self.debug = True  # 是否打印详细信息

# 创建参数实例
params = Parameters()

# 获取跟踪器参数
tracker_params = parameters(params.tracker_param)
tracker_params.debug = params.debug

# 获取数据集
dataset = get_dataset(params.dataset_name)

# 创建跟踪器
tracker = SEQTRACK(tracker_params, params.dataset_name)

# 遍历数据集中的每个序列
for seq_idx, seq in enumerate(dataset):
    print(f"\n=== 处理序列: {seq.name} ===")
    print(f"总帧数: {len(seq.frames)}")
    
    # 初始化跟踪器
    img_files = seq.frames
    anno = seq.ground_truth_rect
    
    # 第一帧作为模板
    init_info = {'init_bbox': anno[0]}
    # 读取第一帧图像
    first_frame = cv2.imread(img_files[0])
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    tracker.initialize(first_frame, init_info)
    
    # 处理后续帧
    for frame_idx in range(1, len(img_files)):
        # 读取当前帧
        frame_path = img_files[frame_idx]
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 跟踪
        output = tracker.track(frame)
        
        # 提取预测结果
        pred_bbox = output['target_bbox']
        confidence = output.get('confidence', None)
        pred_boxes = output.get('pred_boxes', None)
        search_patch = output.get('search_patch', None)
        resize_factor = output.get('resize_factor', None)
        score_maps = output.get('score_maps', None)
        
        # 打印预测信息
        print(f"\n=== 帧 {frame_idx} ===")
        print(f"预测边界框: {pred_bbox}")
        print(f"真实边界框: {anno[frame_idx]}")
        
        # 计算 IoU
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
            
            return intersection / union if union > 0 else 0
        
        iou = calculate_iou(pred_bbox, anno[frame_idx])
        print(f"IoU: {iou:.4f}")
        
        # 打印置信度信息
        if confidence is not None:
            print(f"置信度: {confidence}")
            print(f"平均置信度: {np.mean(confidence):.4f}")
        
        # 打印原始预测框信息
        if pred_boxes is not None:
            print(f"原始预测框: {pred_boxes}")
        
        # 打印搜索区域信息
        if search_patch is not None:
            print(f"搜索区域形状: {search_patch.shape}")
        
        if resize_factor is not None:
            print(f" resize因子: {resize_factor}")
        
        # 打印得分图信息
        if score_maps is not None:
            print(f"得分图形状: {score_maps.shape}")
            # 打印每个坐标的得分图信息
            coordinates = ['x', 'y', 'w', 'h']
            for i, coord in enumerate(coordinates):
                if i < score_maps.shape[1]:
                    score_map = score_maps[0, i]
                    print(f"{coord} 坐标得分图 - 最大值: {np.max(score_map):.4f}, 平均值: {np.mean(score_map):.4f}, 最小值: {np.min(score_map):.4f}")
                    # 找到得分最高的位置
                    max_idx = np.argmax(score_map)
                    print(f"{coord} 坐标得分最高位置: {max_idx}")
        
        # 每处理5帧打印一次进度
        if (frame_idx + 1) % 5 == 0:
            print(f"\n进度: {frame_idx + 1}/{len(img_files)}")
    
    # 打印序列完成信息
    print(f"\n=== 序列 {seq.name} 处理完成 ===")

# 打印所有序列处理完成信息
print(f"\n=== 所有 {len(dataset)} 个序列处理完成 ===")
