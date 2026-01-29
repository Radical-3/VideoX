import cv2
import numpy as np
import os
from lib.test.tracker.seqtrack_utils import sample_target

# 从attention_loss.py中复制convert_bbox_to_crop_coords函数
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

# 测试函数
def test_crop_bbox_verification(image_dir, groundtruth_path):
    # 1. 读取文件夹中的所有图片
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort()  # 按名称排序
    print(f"找到 {len(image_files)} 张图片")
    
    # 2. 读取groundtruth.txt中的所有边界框
    with open(groundtruth_path, 'r') as f:
        groundtruth_lines = f.readlines()
    
    print(f"找到 {len(groundtruth_lines)} 个边界框")
    
    # 3. 确保图片数量和边界框数量一致
    num_tests = min(len(image_files), len(groundtruth_lines))
    
    # 4. 创建保存结果的目录
    output_dir = 'crop_bbox_verification_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 5. 对每张图片进行测试
    for i in range(num_tests):
        print(f"\n=== 测试第 {i+1}/{num_tests} 张图片 ===")
        
        # 5.1 加载图片
        image_file = image_files[i]
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        H, W, _ = image.shape
        print(f"图像: {image_file}, 大小: {W}x{H}")
        
        # 5.2 读取对应的边界框
        line = groundtruth_lines[i].strip()
        # groundtruth.txt的格式是 x1,y1,w,h
        coords = list(map(float, line.split(',')))
        x, y, w, h = coords
        
        # 确保坐标在图像范围内
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(W, x + w)
        y_max = min(H, y + h)
        
        # 重新计算边界框
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        
        original_bbox = [x, y, w, h]
        print(f"原始边界框: {original_bbox}")
        
        # 5.3 模拟跟踪器的裁剪过程
        # 使用与跟踪器相同的参数
        search_factor = 5.0  # 搜索区域因子
        search_size = 256  # 裁剪后的图像大小
        
        # 使用sample_target函数裁剪图像
        crop_image, resize_factor = sample_target(
            image, 
            original_bbox, 
            search_factor, 
            output_sz=search_size
        )
        
        # 5.4 计算裁剪中心和裁剪大小
        # 这部分逻辑来自sample_target函数
        crop_sz = (w * h) ** 0.5 * search_factor
        crop_center = [x + 0.5 * w, y + 0.5 * h]
        print(f"裁剪中心: {crop_center}")
        print(f"裁剪大小: {crop_sz}")
        print(f"Resize因子: {resize_factor}")
        
        # 5.5 使用与adversarial_attack.py相同的方式计算裁剪后的边界框
        crop_bbox = convert_bbox_to_crop_coords(
            original_bbox, 
            crop_center, 
            crop_sz, 
            search_size
        )
        print(f"裁剪后的边界框: {crop_bbox}")
        
        # 5.6 可视化结果
        # 绘制原始图像和边界框
        original_image = image.copy()
        # 绘制原始边界框，确保坐标在图像范围内
        draw_x1 = max(0, int(x))
        draw_y1 = max(0, int(y))
        draw_x2 = min(W, int(x+w))
        draw_y2 = min(H, int(y+h))
        cv2.rectangle(original_image, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
        
        # 绘制裁剪区域
        crop_x1 = crop_center[0] - crop_sz * 0.5
        crop_y1 = crop_center[1] - crop_sz * 0.5
        crop_x2 = crop_x1 + crop_sz
        crop_y2 = crop_y1 + crop_sz
        # 确保裁剪区域坐标在图像范围内
        draw_crop_x1 = max(0, int(crop_x1))
        draw_crop_y1 = max(0, int(crop_y1))
        draw_crop_x2 = min(W, int(crop_x2))
        draw_crop_y2 = min(H, int(crop_y2))
        cv2.rectangle(original_image, (draw_crop_x1, draw_crop_y1), (draw_crop_x2, draw_crop_y2), (255, 0, 0), 2, 1)
        
        # 绘制裁剪后的边界框
        crop_x, crop_y, crop_w, crop_h = crop_bbox
        cv2.rectangle(crop_image, (int(crop_x), int(crop_y)), (int(crop_x+crop_w), int(crop_y+crop_h)), (0, 0, 255), 2)
        
        # 5.7 添加文字说明
        cv2.putText(original_image, 'Original Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(original_image, f'Original BBox: {original_bbox}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(original_image, f'Crop Region: ({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2})', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.putText(crop_image, 'Cropped Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(crop_image, f'Cropped BBox: {crop_bbox}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 5.8 保存结果图像
        # 调整图像大小以便于保存
        original_image_resized = cv2.resize(original_image, (600, 600))
        crop_image_resized = cv2.resize(crop_image, (600, 600))
        
        # 水平拼接两个图像
        combined_image = np.hstack((original_image_resized, crop_image_resized))
        
        # 保存图像
        output_file = os.path.join(output_dir, f'result_{i+1}_{os.path.splitext(image_file)[0]}.png')
        cv2.imwrite(output_file, combined_image)
        print(f"测试图像已保存为: {output_file}")
        
        # 5.9 验证裁剪后的边界框是否正确
        print("\n=== 验证结果 ===")
        # 检查裁剪后的边界框是否在有效范围内
        crop_x, crop_y, crop_w, crop_h = crop_bbox
        if crop_x >= 0 and crop_y >= 0 and crop_x + crop_w <= search_size and crop_y + crop_h <= search_size:
            print("✓ 裁剪后的边界框完全在裁剪图像范围内")
        else:
            print("裁剪后的边界框超出了裁剪图像范围")
        
        # 检查裁剪后的边界框大小是否合理
        expected_w = w * resize_factor
        expected_h = h * resize_factor
        if abs(crop_w - expected_w) < 1 and abs(crop_h - expected_h) < 1:
            print("✓ 裁剪后的边界框大小与预期一致")
        else:
            print("裁剪后的边界框大小与预期不一致")
            print(f"预期: w={expected_w}, h={expected_h}")
            print(f"实际: w={crop_w}, h={crop_h}")

if __name__ == "__main__":
    # 测试目录
    test_image_dir = r"d:\code\VideoX\SeqTrack\data\custom\test\carla_data11_near_img"
    groundtruth_path = r"d:\code\VideoX\SeqTrack\data\custom\test\carla_data11_near_img\groundtruth.txt"
    
    # 运行测试
    test_crop_bbox_verification(test_image_dir, groundtruth_path)
