import os
import cv2
import random
import numpy as np
from glob import glob
from os.path import join

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os.path as path
import pickle
import numpy as np

# region pickle data handling
# def save_pickle(data_dict, file_address):
# 	"""
# 	saves some data in pickle file. data_dict can be a dictionary format
# 	Args:
# 	data_dict: dictionary to save as pickle file
# 	file_address: address of the file to be saved
# 	"""
# 	with open(file_address, 'wb') as handle:
# 		pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# def load_pickle(file_address):
# 	"""
# 	loads a pickle file. file outputs a dictionary
# 	Args:
# 	file_address: address of the .pickle file to be loaded
# 	Returns:
# 	data_dict: dictionary to loaded from pickle file
# 	"""
# 	with open(file_address, 'rb') as handle:
# 		data_dict = pickle.load(handle)
# 	return data_dict
# endregion

class Kitti360Semantic1Hot(Dataset):
	def __init__(self, data_dir:str, sample_size:int, crop_size:int):
		self.data = glob(join(data_dir, '*', 'semantic', '*.png'))
		random.shuffle(self.data)
		self.data = self.data[:sample_size]
		self.crop_size = crop_size
		self.num_classes = 45

	# 	self.data_loaded = None
	#
	# 	self.load_data = False
	# 	if self.load_data == True:
	# 		self.LoadData(data_dir)
	#
	#
	# def LoadData(self, data_dir):
	# 	data_address = join(data_dir, 'data_sampleSize{}_cropSize{}.pickle'.format(len(self.data), self.crop_size))
	# 	if path.exists(data_address):
	# 		self.data_loaded = load_pickle(data_address)
	# 	else:
	# 		self.data_loaded = self.ReadData(data_dir)
	# 		# self.data_loaded = np.asarray(self.data_loaded)
	# 		save_pickle(self.data_loaded, data_address)
	#
	# def ReadData(self, data_dir):
	# 	data_loaded = []
	# 	# data_loaded = np.zeros(shape=(len(self.data), self.crop_size, self.crop_size))
	# 	for index in tqdm(range(len(self.data)), desc="data loading to RAM"):
	# 		# for data_addr in self.data:
	# 		image = cv2.imread(self.data[index])
	# 		image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
	# 		# data_loaded.append(torch.Tensor(image[:,:,0]))
	# 		data_loaded.append(image[:, :, 0])
	# 	return np.asarray(data_loaded)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		# if self.data_loaded is None:
		image = cv2.imread(self.data[index])
		image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
		image = torch.Tensor(image)
		image_semantic_id = image[:, :, 0]
		# else:
		# 	image = self.data_loaded[index]
		# 	image_semantic_id = torch.Tensor(image)


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
		mask_in = torch.cat([background, road, vehicle], dim=0)   # 3xhxw

		# creating the index mask needed for loss calculation
		for i in range(mask_in.shape[0]):
			mask_out += i * mask_in[i]			# hxw

		return {
			"addr": self.data[index],
			# "image": image,
			"mask_in": torch.FloatTensor(mask_in),		# shape  CxHxW      C = number of categories
			"mask_out": torch.FloatTensor(mask_out),	# shape    HxW
			"mask_per_category": {
				"road": road,
				"vehicle": vehicle,
				"background": background,
			}
		}


class Kitti360Semantic(Kitti360Semantic1Hot):
	def __init__(self, data_dir:str, sample_size:int, crop_size:int):
		super(Kitti360Semantic, self).__init__(data_dir, sample_size, crop_size)

	def __getitem__(self, index):
		return_dict = super(Kitti360Semantic, self).__getitem__(index)

		mask = return_dict['mask_out']			# Shape =   HxW

		return {"mask": mask.unsqueeze(0)}		# Shape = 1xHxW


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

if __name__ == "__main__":
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	rcParams['figure.figsize'] = 20, 20

	crop_size = 224
	dataset = Kitti360Semantic1Hot(data_dir="../../Datasets/Kitti360/data_2d_semantics/train", sample_size=10000,
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
		plt.imshow(image_classified['mask_in'].permute(1,2,0))
		plt.title('mask_in', fontsize=25)

		plt.subplot(233)
		plt.imshow(image_classified['mask_out'])
		plt.title('mask_out', fontsize=25)

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

