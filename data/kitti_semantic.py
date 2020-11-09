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

		image_semantic_1hot = torch.zeros(( self.num_classes, image.shape[0], image.shape[1]))	# shape = HxWxC

		for i in range(self.num_classes):
			image_semantic_1hot[i] = torch.where(image_semantic_id == i, ones, zeros)

		# classes determined based on the labels provided by https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
		void_ids = [0, 1, 2, 3, 4, 5, 6, 42, 43, 44]
		flat_ids = [7, 8, 9, 10]
		construction_ids = [11, 12, 13, 14, 15, 16, 34, 35, 36]
		object_ids = [17, 18, 19, 20, 37, 38, 39, 40, 41]
		nature_ids = [21, 22]
		sky_ids = [23]
		human_ids = [24, 25]
		vehicle_ids = [26, 27, 28, 29, 30, 31, 32, 33]

		voids = image_semantic_1hot[void_ids].sum(dim=0, keepdim=True)
		flats = image_semantic_1hot[flat_ids].sum(dim=0, keepdim=True)
		constructions = image_semantic_1hot[construction_ids].sum(dim=0, keepdim=True)
		objects = image_semantic_1hot[object_ids].sum(dim=0, keepdim=True)
		natures = image_semantic_1hot[nature_ids].sum(dim=0, keepdim=True)
		sky = image_semantic_1hot[sky_ids].sum(dim=0, keepdim=True)
		humans = image_semantic_1hot[human_ids].sum(dim=0, keepdim=True)
		vehicles = image_semantic_1hot[vehicle_ids].sum(dim=0, keepdim=True)

		return {
			"addr": self.data[index],
			"mask":
				{
					"voids": voids,
					"flats": flats,
					"constructions": constructions,
					"objects": objects,
					"natures": natures,
					"sky": sky,
					"humans": humans,
					"vehicles": vehicles
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
			"mask": torch.FloatTensor(mask_selected_classes)
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


if __name__ == "__main_categories__":
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	rcParams['figure.figsize'] = 20, 20

	crop_size = 512
	dataset = Kitti360Semantic1Hot(data_dir="../../Datasets/Kitti360/data_2d_semantics/train", sample_size=10,
								   crop_size=crop_size)

	print("len of dataset = {}".format(len(dataset)))

	plot_titles = ['image', 'sky', 'constructions', 'flats', 'vegetation', 'terrain', 'person', 'car']
	for i in range(1):
		image_classified = dataset[np.random.randint(len(dataset))]
		print('data address is ={}'.format(image_classified['addr']))

		plt.subplot(331)
		image = cv2.imread(
			os.path.dirname(image_classified['addr']) + '_rgb/' + os.path.basename(image_classified['addr']))
		image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
		plt.imshow(image)
		plt.title('image', fontsize=25)

		subplot_id = 332
		for key in image_classified['mask'].keys():
			plt.subplot(subplot_id)
			plt.imshow(torch.squeeze(image_classified['mask'][key]))
			plt.title(key, fontsize=25)
			subplot_id += 1
		plt.show()

if __name__ == "__main__":
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	rcParams['figure.figsize'] = 20 ,20

	crop_size = 512
	selected_classes = [23, 7, 8, 21, 22, 24, 26]
	class_titles = ['sky', 'road', 'side walk', 'vegetation', 'terrain', 'person', 'car']
	dataset = Kitti360SemanticAllClasses(data_dir="../../Datasets/Kitti360/data_2d_semantics/train", sample_size=100,
										 crop_size=crop_size, selected_classes= selected_classes)
	print("len of dataset = {}".format(len(dataset)))

	for i in range(1):
		image_classified = dataset[np.random.randint(len(dataset))]
		print('data address is ={}'.format(image_classified['addr']))

		plt.subplot(331)
		image = cv2.imread(os.path.dirname(image_classified['addr']) + '_rgb/' + os.path.basename(image_classified['addr']))
		image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
		plt.imshow(image)
		plt.title('image', fontsize=25)

		subplot_id = 332
		for i, class_title in enumerate(class_titles):
			plt.subplot(subplot_id)
			plt.imshow(image_classified['mask'][i])
			plt.title(class_title, fontsize=25)
			subplot_id += 1
		plt.show()
