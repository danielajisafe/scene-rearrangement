import cv2
import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
import os
import os.path as path
from torch.utils.data import DataLoader
from data.kitti_semantic import Kitti360Semantic1Hot


src_data_path = '../../Datasets/Kitti360/data_2d_semantics/valid'
dest_data_path = '../../Datasets/Kitti360/data_2d_semantics/valid_gt'


crop_size = 256
dataset = Kitti360Semantic1Hot(data_dir=src_data_path, sample_size=None, crop_size=crop_size)
print(f'length of valid dataset = {len(dataset)}')

dataloader = DataLoader( dataset=dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)

data_iter = iter(dataloader)
iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)

for i in iterator:
    batch_data = next(data_iter)

    mask_out = batch_data["mask_out"] # Shape = NxHxW    grayscale
    mask_out = np.asarray(mask_out, dtype=int)
    addr_list = batch_data["addr"]

    for j, addr in enumerate(addr_list):
        dest_file_name = f'{i}_{j}_' + path.basename(addr)[:-4] + '.png'
        dest_file_path = path.join(path.commonpath([src_data_path, addr]) + '_gt', dest_file_name )
        if not path.exists(path.dirname(dest_file_path)):
            os.makedirs(path.dirname(dest_file_path))

        cv2.imwrite(dest_file_path, mask_out[j])