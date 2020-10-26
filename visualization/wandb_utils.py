import wandb
import torch
import numpy as np
import torchvision.utils as vutils


def init_wandb(cfg: dict) -> None:
    """Initialize project on Weights & Biases
    Args:
        cfg (dict): Configuration dictionary
    """
    for key in cfg:
        cfg[key] = cfg[key].__dict__

    wandb.init(project="scene-rearrangement", name=cfg["exp_cfg"]["run_name"], 
        notes=cfg["exp_cfg"]["description"], config=cfg)

def log_epoch_summary(epochID:int, mode:str, losses:dict):
    logs = {}
    for key in losses.keys():
        logs.update({"{}/mean_{}".format(mode, key): losses[key]})

    wandb.log(logs, step=epochID)

def visualize_images(epochID, mode, gt_image, pred_image):
    grid = np.zeros_like(gt_image.cpu().numpy())
    for i in range(0, gt_image.shape[0], 2):
        grid[i] = gt_image[i].cpu().numpy()
        grid[i+1] = pred_image[i].detach().cpu().numpy()

    grid = vutils.make_grid(torch.from_numpy(grid), nrow=2, normalize=True, scale_each=True)

    wandb.log({"{}_reconstruction".format(mode): wandb.Image(grid)}, step=epochID)
