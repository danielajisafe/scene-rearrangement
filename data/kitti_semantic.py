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
		road_ids = [7, 9]
		vehicle_ids = [26, 27, 28, 29, 30, 32, 33]
		background_ids = [i for i in list(range(45)) if (i not in road_ids and i not in vehicle_ids)]


		road = image_semantic_1hot[road_ids].sum(dim=0, keepdim=True)
		vehicle = image_semantic_1hot[vehicle_ids].sum(dim=0, keepdim=True)
		background = image_semantic_1hot[background_ids].sum(dim=0, keepdim=True)

		# fron to back
		mask = torch.cat([vehicle, road, background], dim=0)

		return {
			"addr": self.data[index],
			# "image": image,
			"mask": torch.FloatTensor(mask),
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


if __name__ == "__main__":
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	rcParams['figure.figsize'] = 20, 20

	crop_size = 512
	dataset = Kitti360Semantic1Hot(data_dir="../../Datasets/Kitti360/data_2d_semantics/train", sample_size=10,
								   crop_size=crop_size)

	print("len of dataset = {}".format(len(dataset)))

	for i in range(1):
		image_classified = dataset[np.random.randint(len(dataset))]
		print('data address is ={}'.format(image_classified['addr']))

		plt.subplot(231)
		image = cv2.imread(
			os.path.dirname(image_classified['addr']) + '_rgb/' + os.path.basename(image_classified['addr']))
		image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
		plt.imshow(image)
		plt.title('image', fontsize=25)

		plt.subplot(232)
		plt.imshow(image_classified['mask'].permute(1,2,0))
		plt.title('mask', fontsize=25)

		subplot_id = 234
		for key in image_classified['mask_per_category'].keys():
			plt.subplot(subplot_id)
			plt.imshow(torch.squeeze(image_classified['mask_per_category'][key]))
			plt.title(key, fontsize=25)
			subplot_id += 1
		plt.show()

if __name__ == "__main_AllClasses__":
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
			plt.imshow(image_classified['mask_in'][i])
			plt.title(class_title, fontsize=25)
			subplot_id += 1

		plt.subplot(339)
		plt.imshow(image_classified['mask_out'][0])
		plt.title(class_title, fontsize=25)
		plt.show()

