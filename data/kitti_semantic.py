import os
import cv2
import random
from glob import glob
from os.path import join

import torch
from torch.utils.data import Dataset


class Kitti360Semantic(Dataset):
	def __init__(self, data_dir:str, sample_size:int, crop_size:int):
		self.data = glob(join(data_dir, '*', 'semantic', '*.png'))
		random.shuffle(self.data)
		self.data = self.data[:sample_size]
		self.crop_size = crop_size

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		mask = cv2.imread(self.data[index])
		mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
		mask = mask[:, :, 0:1]

		# Convert to CxHxW
		mask = mask.transpose(2, 0, 1)

		return {"mask": torch.FloatTensor(mask)}


class Kitti360SemanticBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, data_dir: str, crop_size: int, sample_size: int = None, **_ignored):

        self._instance = Kitti360Semantic(
            data_dir=data_dir,
            sample_size=sample_size,
            crop_size=crop_size
        )
        return self._instance
