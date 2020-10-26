import os
import argparse
from os.path import join

from trainer import trainer
from config.config import cfg_parser
from utils.utils import seed_everything


if __name__=="__main__":
    seed_everything(seed=0, harsh=False)
    parser = argparse.ArgumentParser(
        description="Scene Rearrangement Training", allow_abbrev=False
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="vae_test.yml",
        help="name of the config file to use"
        )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
        )
    (args, unknown_args) = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = cfg_parser(join("config", args.version))
    
    pipeline = trainer.factory.create(cfg["model_cfg"].trainer_key, **cfg)
    pipeline.train()