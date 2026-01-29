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
from lib.test.loss.attention_loss import AttentionLoss, get_attention_stats

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
# 启用注意力权重输出
tracker.enable_attention_output()

# 遍历数据集中的每个序列
for seq_idx, seq in enumerate(dataset):
    print(f"\n=== 处理序列: {seq.name} ===")
    print(f"总帧数: {len(seq.frames)}")
    
    # 初始化跟踪器
    img_files = seq.frames
    anno = seq.ground_truth_rect
    
    # 第一帧作为模板
    init_info = {'init_bbox': anno[0]}
    # 读取第一帧图像（npy格式）
    try:
        print(f"尝试加载第一帧: {img_files[0]}")
        first_frame_data = np.load(img_files[0], allow_pickle=True)
        print(f"加载成功，数据类型: {type(first_frame_data)}")
        if hasattr(first_frame_data, 'shape'):
            print(f"数据形状: {first_frame_data.shape}")
        
        # 检查data的类型
        if isinstance(first_frame_data, dict):
            print(f"数据是字典，键: {list(first_frame_data.keys())}")
            # 检查是否有image键
            if 'image' in first_frame_data:
                print(f"找到image键，图像形状: {first_frame_data['image'].shape}")
                first_frame = first_frame_data['image']  # 直接使用npy中的RGB图像
                tracker.initialize(first_frame, init_info)
            else:
                print(f"数据中没有image键")
                continue
        elif first_frame_data.ndim == 0:
            print(f"数据是0维数组，尝试转换为对象")
            first_frame_data = first_frame_data.item()
            print(f"转换后类型: {type(first_frame_data)}")
            if isinstance(first_frame_data, dict):
                print(f"转换后是字典，键: {list(first_frame_data.keys())}")
                if 'image' in first_frame_data:
                    print(f"找到image键，图像形状: {first_frame_data['image'].shape}")
                    first_frame = first_frame_data['image']
                    tracker.initialize(first_frame, init_info)
                else:
                    print(f"数据中没有image键")
                    continue
            else:
                print(f"转换后不是字典")
                continue
        elif first_frame_data.shape == (1,):
            print(f"数据是形状为(1,)的数组，取第一个元素")
            first_frame_data = first_frame_data[0]
            print(f"取元素后类型: {type(first_frame_data)}")
            if isinstance(first_frame_data, dict):
                print(f"取元素后是字典，键: {list(first_frame_data.keys())}")
                if 'image' in first_frame_data:
                    print(f"找到image键，图像形状: {first_frame_data['image'].shape}")
                    first_frame = first_frame_data['image']
                    tracker.initialize(first_frame, init_info)
                else:
                    print(f"数据中没有image键")
                    continue
            else:
                print(f"取元素后不是字典")
                continue
        elif first_frame_data.shape == (6,):
            # 形状为(6,)的数组，顺序: identifier, image, image_mask, label, camera_position, relative_remove
            print(f"数据是形状为(6,)的数组，按顺序提取元素")
            identifier = first_frame_data[0]
            image = first_frame_data[1]
            image_mask = first_frame_data[2]
            label = first_frame_data[3]
            camera_position = first_frame_data[4]
            relative_remove = first_frame_data[5]
            print(f"提取成功，图像形状: {image.shape}")
            print(f"标签: {label}")
            first_frame = image
            tracker.initialize(first_frame, init_info)
        else:
            # 其他情况，使用默认值
            print(f"无法加载第一帧图像: {img_files[0]}")
            continue
    except Exception as e:
        print(f"加载第一帧时出错: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # 处理后续帧
    for frame_idx in range(1, len(img_files)):
        # 读取当前帧（npy格式）
        frame_path = img_files[frame_idx]
        try:
            frame_data = np.load(frame_path, allow_pickle=True)
            # 检查data的类型
            if isinstance(frame_data, dict):
                # 如果直接是字典
                frame = frame_data['image']  # 直接使用npy中的RGB图像
            elif frame_data.ndim == 0:
                # 如果是0维数组，尝试转换为Python对象
                frame_data = frame_data.item()
                if isinstance(frame_data, dict) and 'image' in frame_data:
                    frame = frame_data['image']
                else:
                    print(f"无法加载帧图像: {frame_path}")
                    continue
            elif frame_data.shape == (1,):
                # 如果是形状为(1,)的数组，取第一个元素
                frame_data = frame_data[0]
                if isinstance(frame_data, dict) and 'image' in frame_data:
                    frame = frame_data['image']
                else:
                    print(f"无法加载帧图像: {frame_path}")
                    continue
            elif frame_data.shape == (6,):
                # 形状为(6,)的数组，顺序: identifier, image, image_mask, label, camera_position, relative_remove
                frame = frame_data[1]  # 直接使用第二个元素作为图像
            else:
                # 其他情况，跳过
                print(f"无法加载帧图像: {frame_path}")
                continue
        except Exception as e:
            print(f"加载帧时出错: {e}")
            continue
        
        # 跟踪
        output = tracker.track(frame)
        
        # 提取预测结果
        pred_bbox = output['target_bbox']
        confidence = output.get('confidence', None)
        pred_boxes = output.get('pred_boxes', None)
        search_patch = output.get('search_patch', None)
        resize_factor = output.get('resize_factor', None)
        score_maps = output.get('score_maps', None)
        encoder_attn_weights = output.get('encoder_attn_weights', None)
        cross_attn_weights = output.get('cross_attn_weights', None)
        
        # 打印预测信息
        print(f"\n=== 帧 {frame_idx} ===")
        print(f"预测边界框: {pred_bbox}")
        print(f"真实边界框: {anno[frame_idx]}")
        
        # 检查并转换边界框格式为[x, y, w, h]
        def ensure_bbox_format(bbox):
            if len(bbox) == 4:
                # 检查是否是[x1, y1, x2, y2]格式（通过检查是否x2 > x1和y2 > y1）
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # 转换为[x, y, w, h]格式
                    x = bbox[0]
                    y = bbox[1]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    return [x, y, w, h]
            return bbox
        
        # 确保边界框格式正确
        pred_bbox = ensure_bbox_format(pred_bbox)
        gt_bbox = ensure_bbox_format(anno[frame_idx])
        
        # 计算 IoU
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
            
            return intersection / union if union > 0 else 0
        
        iou = calculate_iou(pred_bbox, gt_bbox)
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
        
        # 打印编码器注意力权重信息
        if encoder_attn_weights is not None:
            print(f"编码器注意力权重层数: {len(encoder_attn_weights)}")
            if encoder_attn_weights:
                # 计算并打印注意力权重统计信息
                attn_stats = get_attention_stats(encoder_attn_weights)
                for layer_name, stats in attn_stats.items():
                    print(f"{layer_name} 注意力统计: 平均值={stats['mean']:.4f}, 最大值={stats['max']:.4f}, 最小值={stats['min']:.4f}, 标准差={stats['std']:.4f}")
        
        # 打印解码器交叉注意力权重信息
        if cross_attn_weights is not None:
            print(f"解码器交叉注意力权重 - 坐标数: {len(cross_attn_weights)}")
            for coord_idx, coord_attn in enumerate(cross_attn_weights):
                print(f"坐标 {['x', 'y', 'w', 'h'][coord_idx]} 的注意力层数: {len(coord_attn)}")
        
        # 每处理5帧打印一次进度
        if (frame_idx + 1) % 5 == 0:
            print(f"\n进度: {frame_idx + 1}/{len(img_files)}")
    
    # 打印序列完成信息
    print(f"\n=== 序列 {seq.name} 处理完成 ===")

# 打印所有序列处理完成信息
print(f"\n=== 所有 {len(dataset)} 个序列处理完成 ===")
