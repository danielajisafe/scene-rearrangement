import os
import argparse
from os.path import join

from config.config import cfg_parser
from utils.utils import seed_everything


if __name__=="__main__":
    seed_everything(seed=0, harsh=False)
    parser = argparse.ArgumentParser(
        description="Mesh Reconstruction Training Zoo", allow_abbrev=False
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="default.yml",
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



    from model.simple_vae import VAE 
    import torch
    import numpy as np 

    xx = torch.FloatTensor(np.random.random((1, 3, 512, 512)))
    vae = VAE(cfg["model_cfg"])
    vae(xx)