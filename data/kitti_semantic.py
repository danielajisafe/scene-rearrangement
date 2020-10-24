import os
import cv2
from glob import glob
from os.path import join

import torch
from torch.utils.data import Dataset


class Kitti360Semantic(Dataset):
	def __init__(self, data_dir:str, sample_size:int, rgb:bool, **ignored):
		if rgb:
			self.data = glob(join(data_dir, '*', 'semantic_rgb', '*.png'))[:sample_size]
		else:
			self.data = glob(join(data_dir, '*', 'semantic', '*.png'))[:sample_size]
		self.rgb = rgb

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		mask = cv2.imread(self.data[index])
		if self.rgb:
			mask = mask/255.
		else:
			mask = mask[:, :, 0:1]

		# Convert to CxHxW
		mask = mask.transpose(2, 0, 1)

		return {"mask": torch.FloatTensor(mask)}
