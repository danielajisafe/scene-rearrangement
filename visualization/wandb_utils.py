import wandb


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