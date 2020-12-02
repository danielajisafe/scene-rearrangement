import os
import wandb
import argparse
from os.path import join, splitext

from trainer import trainer
from utils.logger import set_logger
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
        default="vae-gan-w_3-2-5.yml",
        help="name of the config file to use"
        )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
        )
    parser.add_argument(
        "-w", "--wandb", default=False, action="store_true", help="Log to wandb"
    )
    (args, unknown_args) = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = cfg_parser(join("config", args.version))
    cfg["exp_cfg"].version = splitext(args.version)[0]
    # cfg["exp_cfg"].run_name = cfg["exp_cfg"].version
    cfg["exp_cfg"].wandb = args.wandb

    log_file = join(cfg["exp_cfg"].output_location, cfg["exp_cfg"].version + ".log")
    set_logger(log_file)

    if args.wandb:
        init_wandb(cfg.copy())
    
    pipeline = trainer.factory.create(cfg["model_cfg"].trainer_key, **cfg)
    # pipeline.train()

    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import torch
    import numpy as np
    rcParams['figure.figsize'] = 20, 15
    crop_size = 224

    car_variances = [0, 2, 5, 10, 20, 50]

    subplot = 422
    for i in range(6):
        rearrange_coefficients = [ car_variances[i]/2, car_variances[i]/2, car_variances[i]]
        batch_data, model_out, viz_pred = pipeline.rearrange('val', 0, rearrange_coefficients)

        # np.save('mask_in_10', np.asarray(batch_data['mask_in']))
        #
        # if i==0:
        #     np.save('mask_out_10_good', viz_pred[0])
        # elif i == 2:
        #     np.save('mask_out_10_fair', viz_pred[0])

        print('data address is ={}'.format(batch_data['addr']))

        plt.subplot(421)
        image = cv2.imread(
            os.path.dirname(batch_data['addr']) + '_rgb/' + os.path.basename(batch_data['addr']))
        image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        plt.imshow(image)
        plt.title('image', fontsize=25)

        plt.subplot(422)
        plt.imshow(batch_data['mask_in'].permute(1, 2, 0))
        plt.title('Mask_in', fontsize=25)

        subplot += 1
        plt.subplot(subplot)
        plt.imshow(torch.tensor(viz_pred.squeeze()).permute(1, 2, 0))
        plt.title('rearranged. {}'.format(rearrange_coefficients), fontsize=15)

        # subplot_id = 234
        # for key in batch_data['mask_per_category'].keys():
        #     plt.subplot(subplot_id)
        #     plt.imshow(torch.squeeze(batch_data['mask_per_category'][key]))
        #     plt.title(key, fontsize=25)
        #     subplot_id += 1
    plt.show()
