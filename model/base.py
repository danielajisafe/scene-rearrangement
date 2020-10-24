from torch import nn
from abc import abstractmethod


class BaseModel(nn.Module, ABC):
	def __init__(self):
		super(BaseModel, self).__init__()

		self._build_modules()

	@abstractmethod
	def _build_modules(self):
		pass

	@abstractmethod
	def forward(self):
		pass


class BaseVAE(BaseModel, ABC):
	def __init__(self):
		super(BaseVAE, self).__init__()

	def _build_modules(self):
		self._build_encoder()
		self._build_sampler()
		self._build_decoder()

	@abstractmethod
	def _build_encoder(self):
		pass

	@abstractmethod
	def _build_sampler(self):
		pass

	@abstractmethod
	def _build_decoder(self):
		pass

	@abstractmethod
	def forward(self):
		pass
