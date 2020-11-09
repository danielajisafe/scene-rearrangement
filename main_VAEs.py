import os
import wandb
import argparse
from os.path import join, splitext

from trainer import trainer
from config.config import cfg_parser
from utils.utils import seed_everything
from visualization.wandb_utils import init_wandb


if __name__=="__main__":
    seed_everything(seed=0, harsh=False)
    parser = argparse.ArgumentParser(
        description="Scene Rearrangement Training", allow_abbrev=False
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="vae_per_class.yml",
        help="name of the config file to use"
        )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        default=1,
        help="total number of VAEs to train"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
        )
    parser.add_argument(
        "-w", "--wandb", default=True, action="store_true", help="Log to wandb"
    )

    (args, unknown_args) = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = cfg_parser(join("config", args.version))
    cfg["exp_cfg"].version = splitext(args.version)[0]
    cfg["exp_cfg"].run_name = "experiment_" + cfg["exp_cfg"].version
    cfg["exp_cfg"].wandb = args.wandb

    if args.wandb:
        init_wandb(cfg.copy())

    # training the each stage of VAE that. first stage takes 1 input class.
    for classes in range(args.classes):
        cfg['model_cfg'].network['encoder'][0]['Conv2d']['in_channels'] = classes+1

        pipeline = trainer.factory.create(cfg["model_cfg"].trainer_key, **cfg)
        pipeline.train()