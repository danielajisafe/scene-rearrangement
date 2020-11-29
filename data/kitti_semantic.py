import os
import cv2
import random
import numpy as np
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

class Kitti360Semantic1Hot(Dataset):
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
		image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

		image = torch.Tensor(image)
		image_semantic_id = image[:, :, 0]

		ones = torch.ones(image_semantic_id.shape)
		zeros = torch.zeros(image_semantic_id.shape)

		image_semantic_1hot = torch.zeros(( self.num_classes, image.shape[0], image.shape[1]))	# shape = CxHxW
		mask_out = torch.zeros((image.shape[0], image.shape[1]))	# shape = HxW

		for i in range(self.num_classes):
			image_semantic_1hot[i] = torch.where(image_semantic_id == i, ones, zeros)

		# classes determined based on the labels provided by https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
		road_ids = [7, 9]
		vehicle_ids = [26, 27, 28, 29, 30, 32, 33]
		background_ids = [i for i in list(range(45)) if (i not in road_ids and i not in vehicle_ids)]


		road = image_semantic_1hot[road_ids].sum(dim=0, keepdim=True)
		vehicle = image_semantic_1hot[vehicle_ids].sum(dim=0, keepdim=True)
		background = image_semantic_1hot[background_ids].sum(dim=0, keepdim=True)


		# back to front
		mask_in = torch.cat([background, road, vehicle], dim=0)

		# creating the index mask needed for loss calculation
		for i in range(mask_in.shape[0]):
			mask_out += i * mask_in[i]

		return {
			"addr": self.data[index],
			# "image": image,
			"mask_in": torch.FloatTensor(mask_in),
			"mask_out": torch.FloatTensor(mask_out),
			"mask_per_category": {
				"road": road,
				"vehicle": vehicle,
				"background": background,
			}
		}


class Kitti360Semantic1HotBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, data_dir: str, crop_size: int, sample_size: int = None, **_ignored):

        self._instance = Kitti360Semantic1Hot(
            data_dir=data_dir,
            sample_size=sample_size,
            crop_size=crop_size
        )
        return self._instance



class Kitti360SemanticAllClasses(Dataset):
	def __init__(self, data_dir:str, sample_size:int, crop_size:int, selected_classes:list):
		self.data = glob(join(data_dir, '*', 'semantic', '*.png'))
		random.shuffle(self.data)
		self.data = self.data[:sample_size]
		self.crop_size = crop_size
		self.selected_classes = selected_classes
		self.num_classes = len(self.selected_classes)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		image = cv2.imread(self.data[index])
		image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

		image = torch.Tensor(image)
		image_semantic_id = image[:, :, 0]

		ones = torch.ones(image_semantic_id.shape)
		zeros = torch.zeros(image_semantic_id.shape)

		mask_selected_classes = torch.zeros(( self.num_classes, image.shape[0], image.shape[1]))	# shape = HxWxC

		# for i in range(self.num_classes):
		# 	classes.append(torch.where(image_semantic_id == i, ones, zeros))
		for i, selected_class in enumerate(self.selected_classes):
			mask_selected_classes[i] = torch.where(image_semantic_id == selected_class, ones, zeros)

		return {
			"addr": self.data[index],
			"mask_in": torch.FloatTensor(mask_selected_classes),
			"mask_out": torch.FloatTensor(mask_selected_classes[-1:, :, :])
		}


class Kitti360SemanticAllClassesBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, data_dir: str, crop_size: int, sample_size: int = None, selected_classes: list = None, **_ignored):

        self._instance = Kitti360SemanticAllClasses(
            data_dir=data_dir,
            sample_size=sample_size,
            crop_size=crop_size,
			selected_classes=selected_classes,
        )
        return self._instance

class Kitti360Semantic1HotAdv(Dataset):
	def __init__(self, data_dir:str, sample_size:int, crop_size:int):
		self.data = glob(join(data_dir, '*', 'semantic', '*.png'))
		random.shuffle(self.data)
		self.data = self.data[:sample_size]
		self.crop_size = crop_size
		self.num_classes = 45

		# classes determined based on the labels provided by https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
		self.road_ids = [7, 9]
		self.vehicle_ids = [26, 27, 28, 29, 30, 32, 33]
		self.background_ids = [i for i in list(range(45)) if (i not in self.road_ids and i not in self.vehicle_ids)]

	def __len__(self):
		return len(self.data)

	def get_processed(self, image):
		image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

		image = torch.Tensor(image)
		image_semantic_id = image[:, :, 0]

		ones = torch.ones(image_semantic_id.shape)
		zeros = torch.zeros(image_semantic_id.shape)

		image_semantic_1hot = torch.zeros(( self.num_classes, image.shape[0], image.shape[1]))	# shape = CxHxW
		mask_out = torch.zeros((image.shape[0], image.shape[1]))	# shape = HxW

		for i in range(self.num_classes):
			image_semantic_1hot[i] = torch.where(image_semantic_id == i, ones, zeros)


		road = image_semantic_1hot[self.road_ids].sum(dim=0, keepdim=True)
		vehicle = image_semantic_1hot[self.vehicle_ids].sum(dim=0, keepdim=True)
		background = image_semantic_1hot[self.background_ids].sum(dim=0, keepdim=True)

		# back to front
		mask_in = torch.cat([background, road, vehicle], dim=0)

		# creating the index mask needed for loss calculation
		for i in range(mask_in.shape[0]):
			mask_out += i * mask_in[i]

		return mask_in, mask_out, road, vehicle, background

	def __getitem__(self, index):
		image = cv2.imread(self.data[index])
		mask_in, mask_out, road, vehicle, background = self.get_processed(image)

		adv = cv2.imread(self.data[np.random.randint(self.__len__())])
		adv_mask, _, _, _, _ = self.get_processed(adv)

		return {
			"addr": self.data[index],
			# "image": image,
			"mask_in": torch.FloatTensor(mask_in),
			"mask_out": torch.FloatTensor(mask_out),
			"mask_per_category": {
				"road": road,
				"vehicle": vehicle,
				"background": background,
			},
			"adv_mask": torch.FloatTensor(adv_mask)
		}


class Kitti360Semantic1HotAdvBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, data_dir: str, crop_size: int, sample_size: int = None, **_ignored):

        self._instance = Kitti360Semantic1HotAdv(
            data_dir=data_dir,
            sample_size=sample_size,
            crop_size=crop_size
        )
        return self._instance


