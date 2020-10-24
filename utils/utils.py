import torch
import random
import numpy as np


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
