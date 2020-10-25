from tqdm import tqdm
from collections import defaultdict, OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import data
from model import model
from utils.utils import dict_to_device
from losses.losses import mse, KL


class VAETrainer(object):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_dataloaders()
        self.model = model.factory.create(self.model_cfg.model_key, **{"model_cfg": self.model_cfg}).to(self.device)
        self._setup_optimizers()

    def _setup_dataloaders(self):
        self.datasets = defaultdict(list)
        for mode in self.model_cfg.modes:
            for source in self.data_cfg.sources:
                for key, args in source.items():
                    args.update({"data_dir": args[mode]})
                    self.datasets[mode].append((data.factory.create(key, **args), args["batch_size"], key))

        self.dataloaders = self.create_dataloaders()

    def create_dataloaders(self):
        dataloaders = defaultdict(OrderedDict)

        for mode in self.datasets.keys():

            shuffle = mode == "train"
            drop_last = mode == "train"

            for dataset in self.datasets[mode]:
                dataloader = DataLoader(
                    dataset=dataset[0],
                    batch_size=dataset[1],
                    shuffle=shuffle,
                    drop_last=drop_last,
                    pin_memory=True,
                    num_workers=self.data_cfg.num_workers,
                )

                dataloaders[mode][dataset[2]] = dataloader

        return dataloaders

    def _setup_optimizers(self):
        vae_params = list(self.model.parameters())
        vae_optim_cfg = self.model_cfg.optimizers["vae"]

        self.vae_opt =  eval(
            "optim.{}(vae_params, **{})".format([*vae_optim_cfg.keys()][0], [*vae_optim_cfg.values()][0])
        )

    def _backprop(self, loss):
        self.vae_opt.zero_grad()
        loss.backward()
        self.vae_opt.step()

    def _epoch(self, mode:str, epochID: int):
        if mode.__eq__('train'):
            self.model.train()
        else:
            self.model.eval()
        data_iter = iter(self.dataloaders[mode]["kitti360_semantic"])
        iterator = tqdm(range(len(self.dataloaders[mode]["kitti360_semantic"])), dynamic_ncols=True)

        for i in iterator:
            batch_data = dict_to_device(next(data_iter), self.device)
            model_out = self.model(batch_data["mask"])

            reconst_loss = mse(model_out.reconst, batch_data['mask'])
            kld = KL(model_out.mu, model_out.log_var)

            loss = reconst_loss + kld

            self._backprop(loss)

            iterator.set_description("V: {} | Epoch: {} | {} | Loss: {:.4f}".format(self.exp_cfg.cfg_file,
                epochID, mode, loss.item()), refresh=True)

    def train(self):
        for epochID in range(self.model_cfg.epochs):
            for mode in self.model_cfg.modes:
                self._epoch(mode, epochID)


class VAETrainerBuilder(object):
    """VAE Trainer Builder Class
    """

    def __init__(self):
        """VAE Trainer Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            VAETrainer: Instantiated VAE trainer object
        """
        if not self._instance:
            self._instance = VAETrainer(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
