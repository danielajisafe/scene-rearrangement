import torch
from torch import nn
from collections import namedtuple

from model.base import BaseVAE
from model.network import Network


class VAE(BaseVAE):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(VAE, self).__init__()

    def _build_encoder(self):
        # TODO make the encoder layers here
        self.encoder = Network(self.cfg.network["encoder"])
        self.encoder_fc = Network(self.cfg.network["encoder_fc"])
        self.fc_mu = Network(self.cfg.network["fc_mu"])
        self.fc_var = Network(self.cfg.network["fc_var"])

    def _build_decoder(self):
        # TODO make the decoder layers here
        self.decoder_fc = Network(self.cfg.network["decoder_fc"])
        self.decoder = Network(self.cfg.network["decoder"])

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.encoder_fc(result)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

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

    def decode(self, z):
        result = self.decoder_fc(z)
        # result = result.view(-1, 512, 2, 2)
        result = result.view(-1, 128, 2, 2)         # TODO fix this part to be nicer :)
        result = self.decoder(result)

        return result

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        model_out_tuple = namedtuple(
            "model_out",
            ["reconst", "mu", "log_var"],
        )
        model_out = model_out_tuple(
            self.decode(z), mu, log_var
        )

        return model_out

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.cfg.latent_dim)
        z = z.to(self.device)

        samples = self.decode(z)
        return samples


class VAEBuilder(object):
    """VAE Model Builder Class
    """

    def __init__(self):
        """VAE Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            VAE: Instantiated VAE network object
        """
        if not self._instance:
            self._instance = VAE(model_cfg=model_cfg)
        return self._instance
