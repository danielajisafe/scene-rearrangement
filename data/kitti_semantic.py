import os
import cv2
from glob import glob
from os.path import join

import torch
from torch.utils.data import Dataset


class Kitti360Semantic(Dataset):
	def __init__(self, data_dir:str, sample_size:int, rgb:bool, crop_size:int):
		if rgb:
			self.data = glob(join(data_dir, '*', 'semantic_rgb', '*.png'))[:sample_size]
		else:
			self.data = glob(join(data_dir, '*', 'semantic', '*.png'))[:sample_size]
		self.rgb = rgb
		self.crop_size = crop_size

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		mask = cv2.imread(self.data[index])
		mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
		if self.rgb:
			mask = mask/255.
		else:
			mask = mask[:, :, 0:1]/255.

		# Convert to CxHxW
		mask = mask.transpose(2, 0, 1)

		return {"mask": torch.FloatTensor(mask)}


class Kitti360SemanticBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, rgb: bool, data_dir: str, crop_size: int, sample_size: int = None, **_ignored):

        self._instance = Kitti360Semantic(
            data_dir=data_dir,
            sample_size=sample_size,
            rgb=rgb,
            crop_size=crop_size
        )
        return self._instance
