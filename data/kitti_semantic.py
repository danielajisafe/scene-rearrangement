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

class Kitti360Classes(Dataset):
	def __init__(self, data_dir:str, sample_size:int, crop_size:int):
		self.data = glob(join(data_dir, '*', 'semantic', '*.png'))
		random.shuffle(self.data)
		self.data = self.data[:sample_size]
		self.crop_size = crop_size
		self.num_classes = 45

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		image = cv2.imread(self.data[index])
		image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)

		image = torch.Tensor(image)

		image = torch.mean(image, dim=-1)

		ones = torch.ones(image.shape)
		# ones = torch.ones((image.shape[0], image.shape[1], self.num_classes))
		zeros = torch.zeros(image.shape)
		# zeros = torch.zeros((image.shape[0], image.shape[1], self.num_classes))

		binary_classification = torch.ones((self.num_classes, image.shape[0], image.shape[1]))

		for i in range(self.num_classes):
			binary_classification[i] = torch.where(image==i, ones, zeros)


		# classes determined based on the labels provided by https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
		void_ids = [0, 1, 2, 3, 4, 5, 6, 42, 43, 44]
		flat_ids = [7, 8, 9, 10]
		construction_ids = [11, 12, 13, 14, 15, 16, 34, 35, 36]
		object_ids = [17, 18, 19, 20, 21, 22, 37, 38, 39, 40, 41]
		nature_ids = [21, 22]
		sky_ids = [23]
		human_ids = [24, 25]
		vehicle_ids = [26, 27, 28, 29, 30, 31, 32, 33]

		voids = torch.unsqueeze(binary_classification[void_ids].sum(dim=0), dim=0)
		flats = torch.unsqueeze(binary_classification[flat_ids].sum(dim=0), dim=0)
		constructions = torch.unsqueeze(binary_classification[construction_ids].sum(dim=0), dim=0)
		objects = torch.unsqueeze(binary_classification[object_ids].sum(dim=0), dim=0)
		natures = torch.unsqueeze(binary_classification[nature_ids].sum(dim=0), dim=0)
		sky = torch.unsqueeze(binary_classification[sky_ids].sum(dim=0), dim=0)
		humans = torch.unsqueeze(binary_classification[human_ids].sum(dim=0), dim=0)
		vehicles = torch.unsqueeze(binary_classification[vehicle_ids].sum(dim=0), dim=0)

		return {
			"voids": voids,
			"flats": flats,
			"constructions": constructions,
			"objects": objects,
			"natures": natures,
			"sky": sky,
			"humans": humans,
			"vehicles": vehicles
		}


class Kitti360ClassesBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, data_dir: str, crop_size: int, sample_size: int = None, **_ignored):

        self._instance = Kitti360Classes(
            data_dir=data_dir,
            sample_size=sample_size,
            crop_size=crop_size
        )
        return self._instance