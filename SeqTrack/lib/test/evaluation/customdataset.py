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
        # 构建标注路径
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        
        # 加载标注
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        
        # 构建帧路径
        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        # 获取所有帧文件
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg") or frame.endswith(".png")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        
        # 返回序列对象
        return Sequence(sequence_name, frames_list, 'custom', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        # 从list.txt加载序列列表
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
