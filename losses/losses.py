import torch
from torch.nn import functional as F


def mse(x, y):
	return F.mse_loss(x, y)

def KL(mu, log_var):
	'''
	KL divergence between N(mu, sigma) and N(0, 1)
	'''
	return torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)