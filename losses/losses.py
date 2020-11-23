import torch
from torch.nn import functional as F


def mse(x, y):
	'''
	x: network output of shape NCHW
	y: ground truth image with shape NCHW
	'''
	return F.mse_loss(x, y)

def mae(x, y):
	'''
	x: network output of shape NCHW
	y: ground truth image with shape NCHW
	'''
	return F.l1_loss(x, y)

def cross_entropy(x, y, weights):
	'''
	x: network output of shape (N, num_classes, H ,W)
	y: ground truth mask with shape (N, H, W)
	'''
	return F.cross_entropy(x, y.long(), weight=torch.FloatTensor(weights).to(x.device))

def KL(mu, log_var):
	'''
	KL divergence between N(mu, sigma) and N(0, 1)
	'''
	return torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

def elbo(mu, log_var, z, reconst, input):
	# TODO
	pass
