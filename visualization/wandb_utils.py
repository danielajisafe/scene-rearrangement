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
    grid = np.zeros((8, gt_image.shape[1], gt_image.shape[2], gt_image.shape[3]))

    if pred_image.shape[1] != gt_image.shape[1]:
        pred_image = np.expand_dims(pred_image.argmax(axis=1), 1)

    for i in range(grid.shape[0]//2):
        grid[i] = gt_image[i]
        grid[i+4] = pred_image[i]

    grid = vutils.make_grid(torch.from_numpy(grid), nrow=4, normalize=True, scale_each=True)

    ## Draw stage-wise predictions. Currently hardcoded to 3 classes.
    # bg = np.zeros((4, 3, pred_image.shape[2], pred_image.shape[3]))
    # road = np.zeros((4, 3, pred_image.shape[2], pred_image.shape[3]))
    # cars = np.zeros((4, 3, pred_image.shape[2], pred_image.shape[3]))

    # bg[:, 0:1] = pred_image[:4, 0:1]
    # road[:, 1:2] = pred_image[:4, 1:2]
    # cars[:, 2:3] = pred_image[:4, 2:3]

    # grid_layers = np.zeros((12, 3, pred_image.shape[2], pred_image.shape[3]))
    # for i in range(4):
    #     grid_layers[i] = bg[i]
    #     grid_layers[i+4] = road[i]
    #     grid_layers[i+8] = cars[i]

    # grid_layers = vutils.make_grid(torch.from_numpy(grid_layers), nrow=4, normalize=True, scale_each=True)
    # wandb.log({"{}_reconstruction".format(mode): wandb.Image(torch.cat([grid, grid_layers], dim=1))}, step=epochID)
    wandb.log({"{}_reconstruction".format(mode): wandb.Image(grid)}, step=epochID)

def visualize_GuidedRearranged_Images(epochID, mode, gt_image, pred_image, rearranged_image):
    grid = np.zeros((12, gt_image.shape[1], gt_image.shape[2], gt_image.shape[3]))

    if pred_image.shape[1] != gt_image.shape[1]:
        pred_image = np.expand_dims(pred_image.argmax(axis=1), 1)

    if rearranged_image.shape[1] != gt_image.shape[1]:
        rearranged_image = np.expand_dims(rearranged_image.argmax(axis=1), 1)

    for i in range(grid.shape[0]//3):
        grid[i] = gt_image[i]
        grid[i+4] = pred_image[i]
        grid[i+8] = rearranged_image[i]

    grid = vutils.make_grid(torch.from_numpy(grid), nrow=4, normalize=True, scale_each=True)

    wandb.log({"{}_reconstruction_rearranged".format(mode): wandb.Image(grid)}, step=epochID)
