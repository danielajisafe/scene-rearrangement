import torch
from torch import nn
import kornia.geometry as K
from torchvision import models
from torchvision.transforms import RandomAffine
from collections import namedtuple, defaultdict

from model.base import BaseVAE
from model.network import Network
from utils.utils import StraightThroughEstimator


class ShiftGAN(BaseVAE):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(ShiftGAN, self).__init__()
        self._init_vaes()
        self._build_discriminator()
        self.STE = StraightThroughEstimator()

    def _build_encoder(self):
        pass

    def _build_decoder(self):
        pass

    def _build_discriminator(self):
        self.discriminator = Network(self.cfg.network["discriminator"])
        self.discriminator_fc = Network(self.cfg.network["discriminator_fc"])

    def encode(self, idx, input):
        '''
        idx (int): stage of the VAE
        input (torch.Tensor): (N, 1, H, W)
        '''
        result = self.encoders_pre[idx](input)
        if idx == 0:
            result = torch.cat([result, torch.zeros_like(result)], dim=1)
        else:
            result = torch.cat([result, self.decoder_intermediates[idx - 1]], dim=1)
        result = self.encoders_post[idx](result)
        result = torch.flatten(result, start_dim=1)
        result = self.encoders_fc[idx](result)

        mu = self.fc_mus[idx](result)
        log_var = self.fc_vars[idx](result)

        shift_mu = self.shift_mus[idx](result)
        shift_logvar = self.shift_vars[idx](result)

        return mu, log_var, shift_mu, shift_logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, idx, z):
        '''
        idx (int): stage of the VAE
        '''
        result = self.decoders_fc[idx](z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoders_pre[idx](result)
        self.decoder_intermediates.append(result)
        result = self.decoders_post[idx](result)

        return result

    def _init_vaes(self):
        self.encoders_pre = nn.ModuleList(
                [
                    Network(self.cfg.network['encoder_pre']) for i in range(self.cfg.n_classes)
                ]
            )
        self.encoders_post = nn.ModuleList(
                [
                    Network(self.cfg.network['encoder_post']) for i in range(self.cfg.n_classes)
                ]
            )
        self.encoders_fc = nn.ModuleList(
                [
                    Network(self.cfg.network['encoder_fc']) for i in range(self.cfg.n_classes)
                ]
            )
        self.fc_mus = nn.ModuleList(
                [
                    Network(self.cfg.network['fc_mu']) for i in range(self.cfg.n_classes)
                ]
            )
        self.fc_vars = nn.ModuleList(
                [
                    Network(self.cfg.network['fc_var']) for i in range(self.cfg.n_classes)
                ]
            )
        self.shift_mus = nn.ModuleList(
                [
                    Network(self.cfg.network['shift_mu']) for i in range(self.cfg.n_classes)
                ]
            )
        self.shift_vars = nn.ModuleList(
                [
                    Network(self.cfg.network['shift_var']) for i in range(self.cfg.n_classes)
                ]
            )
        self.decoders_fc = nn.ModuleList(
                [
                    Network(self.cfg.network['decoder_fc']) for i in range(self.cfg.n_classes)
                ]
            )
        self.decoders_pre = nn.ModuleList(
                [
                    Network(self.cfg.network['decoder_pre']) for i in range(self.cfg.n_classes)
                ]
            )
        self.decoders_post = nn.ModuleList(
                [
                    Network(self.cfg.network['decoder_post']) for i in range(self.cfg.n_classes)
                ]
            )

    def shift_img(self, stage, decoded, shift_mu, shift_logvar):
        if stage not in self.cfg.shift_classes:
            shifts = torch.zeros((decoded.shape[0], 2)).to(self.device)
        else:
            N, C, H, W = decoded.shape
            shifts = self.reparameterize(shift_mu, shift_logvar)
            shifts = shifts * torch.tensor([W, H]).to(self.device)

        shifted = K.translate(decoded, shifts)
        return shifted
        
    def forward(self, input):
        '''
        input: (N, n_classes, H, W)
        '''
        self.decoder_intermediates = list()
        vae_outputs = defaultdict(list)
        for stage in range(self.cfg.n_classes):
            mu, log_var, shift_mu, shift_logvar = self.encode(stage, input[:, stage:stage+1])
            z = self.reparameterize(mu, log_var)
            decoded = self.decode(stage, z)
            shifted = self.shift_img(stage, decoded, shift_mu, shift_logvar)

            vae_outputs['mu'].append(mu)
            vae_outputs['log_var'].append(log_var)
            vae_outputs['decoded'].append(decoded)
            vae_outputs['shifted'].append(shifted)

        vae_outputs['decoded'] = torch.cat(vae_outputs['decoded'], dim=1)
        vae_outputs['shifted'] = torch.cat(vae_outputs['shifted'], dim=1)
        model_out_tuple = namedtuple(
            "model_out", vae_outputs
        )
        model_out = model_out_tuple(
            **vae_outputs
        )

        return model_out

    def disc_forward(self, x):
        x = self.discriminator(x)
        x = torch.flatten(x, start_dim=1)
        x = self.discriminator_fc(x)
        return x

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.cfg.latent_dim)
        z = z.to(self.device)

        samples = self.decode(z)
        return samples


class ShiftGANBuilder(object):
    """ShiftGAN Model Builder Class
    """

    def __init__(self):
        """ShiftGAN Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            ShiftGAN: Instantiated ShiftGAN network object
        """
        if not self._instance:
            self._instance = ShiftGAN(model_cfg=model_cfg)
        return self._instance
