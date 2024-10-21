import os
import random
from scipy.io import loadmat
from collections import defaultdict

from .utils import Datum_describle, DatasetBase, read_json, read_json_utf, Datum


class RSTPReid(DatasetBase):
    dataset_dir = 'RSTPReid'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.label_file = os.path.join(self.dataset_dir, 'imagelabels.mat')
        self.lab2cname_file = os.path.join(self.dataset_dir, 'cat_to_name.json')

        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_RSTPReid.json')

        # 它分数据集是跟着我的文件分的
        train, val, test = self.read_split_utf(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_split_utf(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label,description in items:
                impath = os.path.join(path_prefix, impath)
                description1, description2 = description
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=description1

                )
                out.append(item)
            return out

        print(f'Reading split from {filepath}')
        split = read_json_utf(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test