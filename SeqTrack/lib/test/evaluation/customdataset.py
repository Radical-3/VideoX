import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class CustomDataset(BaseDataset):
    """自定义数据集"""
    def __init__(self, split):
        super().__init__()
        # 设置数据集路径
        self.base_path = os.path.join(self.env_settings.custom_path, split)
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # 构建帧路径
        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        # 获取所有帧文件
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".npy")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        
        # 从npy文件中加载标注
        ground_truth_rect = []
        for frame_path in frames_list:
            try:
                # 尝试不同的加载方式
                data = np.load(frame_path, allow_pickle=True)
                # 检查data的类型
                if isinstance(data, dict):
                    # 如果直接是字典
                    # 提取label作为真实边界框
                    if 'label' in data:
                        ground_truth_rect.append(data['label'])
                    else:
                        # 如果没有label，使用默认值
                        ground_truth_rect.append([0, 0, 0, 0])
                elif data.ndim == 0:
                    # 如果是0维数组，尝试转换为Python对象
                    data = data.item()
                    if isinstance(data, dict) and 'label' in data:
                        ground_truth_rect.append(data['label'])
                    else:
                        ground_truth_rect.append([0, 0, 0, 0])
                elif data.shape == (1,):
                    # 如果是形状为(1,)的数组，取第一个元素
                    data = data[0]
                    if isinstance(data, dict) and 'label' in data:
                        ground_truth_rect.append(data['label'])
                    else:
                        ground_truth_rect.append([0, 0, 0, 0])
                elif data.shape == (6,):
                    # 形状为(6,)的数组，顺序: identifier, image, image_mask, label, camera_position, relative_remove
                    label = data[3]  # 第四个元素是label
                    # 检查边界框格式并转换为[x, y, w, h]
                    if len(label) == 4:
                        # 检查是否是[x1, y1, x2, y2]格式（通过检查是否x2 > x1和y2 > y1）
                        if label[2] > label[0] and label[3] > label[1]:
                            # 转换为[x, y, w, h]格式
                            x = label[0]
                            y = label[1]
                            w = label[2] - label[0]
                            h = label[3] - label[1]
                            label = [x, y, w, h]
                    ground_truth_rect.append(label)
                else:
                    # 其他情况，使用默认值
                    ground_truth_rect.append([0, 0, 0, 0])
            except Exception as e:
                print(f"加载文件 {frame_path} 时出错: {e}")
                # 出错时使用默认值
                ground_truth_rect.append([0, 0, 0, 0])
        ground_truth_rect = np.array(ground_truth_rect, dtype=np.float64)
        
        # 返回序列对象
        return Sequence(sequence_name, frames_list, 'custom', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        # 从list.txt加载序列列表
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
