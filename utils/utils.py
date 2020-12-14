import torch
import random
import numpy as np
import logging
from PIL import Image
import cv2
import os.path as path
import os
import torchvision

def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def dict_to_device(d_ten: dict, device):
    """
    Sets a dictionary to device
    Args:
        d_ten (dict): dictionary of tensors
        device (str): torch device
    Returns:
        dict: dictionary on device
    """
    for key, tensor in d_ten.items():
        if type(tensor) is torch.Tensor:
            d_ten[key] = d_ten[key].to(device)

    return d_ten

def detach_2_np(x: torch.tensor):
    return x.detach().cpu().numpy()
    
def copy_state_dict(cur_state_dict, pre_state_dict, prefix=""):
    """
        Load parameters
    Args:
        cur_state_dict (dict): current parameters
        pre_state_dict ([type]): load parameters
        prefix (str, optional): specific module names. Defaults to "".
    """

    def _get_params(key):
        key = prefix + key
        try:
            out = pre_state_dict[key]
        except Exception:
            try:
                out = pre_state_dict[key[7:]]
            except Exception:
                try:
                    out = pre_state_dict["module." + key]
                except Exception:
                    try:
                        out = pre_state_dict[key[14:]]
                    except Exception:
                        out = None
        return out

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                logging.info("parameter {} not found".format(k))
                continue
            cur_state_dict[k].copy_(v)
        except Exception:
            logging.info("copy param {} failed".format(k))
            continue


def write_images(batch_num, this_batch, folder, addr_list, dest_data_path):
    # dest_data_path = '../../Datasets/Kitti360/data_2d_semantics/valid_multistage'
    mask_out = this_batch.cpu().numpy()
    out_image = np.zeros((len(mask_out), mask_out.shape[2], mask_out.shape[3]))
    for i in range(mask_out.shape[1]):
        out_image += i * mask_out[:, i]

    out_image = np.moveaxis(out_image, [0,1,2], [2,0,1])
    out_image = cv2.resize(out_image, (256, 256), interpolation=cv2.INTER_NEAREST)

    # out_image_int = out_image.ceil().astype(int)

    # this_batch = torchvision.transforms.Resize((256, 256), interpolation=Image.NEAREST)(this_batch).cpu().numpy()
    # # creating the grayscale segmasks in which each pixel holds the class index
    # out_image = np.zeros((len(this_batch), 256, 256))
    # for i in range(this_batch.shape[1]):
    #     out_image += i * this_batch[:, i]

    #Saving the images as png files
    for j, addr in enumerate(addr_list):
        dest_file_name = f'{batch_num}_{j}_' + path.basename(addr)[:-4] + '.png'
        dest_file_path = path.join(dest_data_path, folder, dest_file_name)
        if not path.exists(path.dirname(dest_file_path)):
            os.makedirs(path.dirname(dest_file_path))

        cv2.imwrite(dest_file_path, out_image[:,:,j])
