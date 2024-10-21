import os
import random
from scipy.io import loadmat
from collections import defaultdict

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, read_json

template = ['a photo of a {}, a type of flower.']


class test(DatasetBase):
    dataset_dir = 'test'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.lab2cname_file = os.path.join(self.dataset_dir, 'cat_to_name.json')
        self.template = template
        super().__init__(train_x=test)

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)['labels'][0]
        for i, label in enumerate(label_file):
            imname = f'image_{str(i + 1).zfill(5)}.jpg'
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print('Splitting data into 50% train, 20% val, and 30% test')

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y - 1,  # convert to 0-based label
                    classname=c
                )
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        test = []
        for label, impaths in tracker.items():

            cname = lab2cname[str(label)]
            test.extend(_collate(impaths, label, cname))

        return test