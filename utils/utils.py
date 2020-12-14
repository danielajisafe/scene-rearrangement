import os
import cv2
import torch
import random
import logging
import numpy as np
from os.path import join, basename


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)

class StraightThroughEstimator(torch.nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


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

def dump_model_output(output, mode, outdir, fname, flag='reconstructed'):
    outdir = join(outdir, 'outputs', mode, flag)
    os.makedirs(outdir, exist_ok=True)
    output = detach_2_np(torch.argmax(output, dim=1))
    for i, x in enumerate(output):
        path = join(outdir, '_'.join(fname[i].split('/')[-3::2]))
        cv2.imwrite(path, cv2.resize(np.array(x, dtype=np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST))
