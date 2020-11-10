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
		

		return {
			"addr": self.data[index],
			# "image": image,
			"road": road,
			"vehicle": vehicle,
			"background": background,
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

		classes = []

		for i in range(self.num_classes):
			classes.append(torch.where(image_semantic_id == i, ones, zeros))

		return classes


class Kitti360SemanticAllClassesBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, data_dir: str, crop_size: int, sample_size: int = None, **_ignored):

        self._instance = Kitti360SemanticAllClasses(
            data_dir=data_dir,
            sample_size=sample_size,
            crop_size=crop_size
        )
        return self._instance


if __name__ == "__main__":
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	rcParams['figure.figsize'] = 20 ,20

	crop_size = 512
	dataset = Kitti360Semantic1Hot(data_dir="../../Datasets/Kitti360/data_2d_semantics/train", sample_size=10, crop_size=crop_size)

	print("len of dataset = {}".format(len(dataset)))

	for i in range(1):
		image_classified = dataset[np.random.randint(len(dataset))]
		print('data address is ={}'.format(image_classified['addr']))

		plt.subplot(331)
		image = cv2.imread(os.path.dirname(image_classified['addr']) + '_rgb/' + os.path.basename(image_classified['addr']))
		image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
		plt.imshow(image)
		plt.title('image', fontsize=25)

		plt.subplot(332)
		plt.imshow(torch.squeeze(image_classified['sky']))
		plt.title('sky', fontsize=25)

		plt.subplot(333)
		plt.imshow(torch.squeeze(image_classified['constructions']))
		plt.title('constructions', fontsize=25)

		plt.subplot(334)
		plt.imshow(torch.squeeze(image_classified['flats']))
		plt.title('flats', fontsize=25)

		plt.subplot(335)
		plt.imshow(torch.squeeze(image_classified['natures']))
		plt.title('natures', fontsize=25)

		plt.subplot(336)
		plt.imshow(torch.squeeze(image_classified['vehicles']))
		plt.title('vehicles', fontsize=25)

		plt.subplot(337)
		plt.imshow(torch.squeeze(image_classified['humans']))
		plt.title('humans', fontsize=25)

		plt.subplot(338)
		plt.imshow(torch.squeeze(image_classified['objects']))
		plt.title('objects', fontsize=25)

		plt.subplot(339)
		plt.imshow(torch.squeeze(image_classified['voids']))
		plt.title('voids', fontsize=25)
		plt.show()
