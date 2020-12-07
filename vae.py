import argparse
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import cv2
from tqdm import tqdm
from glob import glob
import random
from os.path import join

import wandb

class Kitti360Semantic1Hot(Dataset):
   def __init__(self, data_dir:str, crop_size:int):
      self.data = glob(join(data_dir, '*', 'semantic', '*.png'))

      self.crop_size = crop_size

      self.num_classes = 45

      self.road_ids = [7, 9]
      self.vehicle_ids = [26, 27, 28, 29, 30, 32, 33]

   def __len__(self):
      return len(self.data)

   def __getitem__(self, index):
      # if self.data_loaded is None:
      image = cv2.imread(self.data[index])
      image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
      image = torch.Tensor(image)
      image_semantic_id = image[:, :, 0]


      ones = torch.ones(image_semantic_id.shape)
      zeros = torch.zeros(image_semantic_id.shape)

      image_semantic_1hot = torch.zeros(( self.num_classes, image.shape[0], image.shape[1]))	# shape = CxHxW
      mask_out = torch.zeros((image.shape[0], image.shape[1]))	# shape = HxW

      for i in self.road_ids+self.vehicle_ids:
         image_semantic_1hot[i] = torch.where(image_semantic_id == i, ones, zeros)

      # classes determined based on the labels provided by https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py

      road = image_semantic_1hot[self.road_ids].sum(dim=0, keepdim=True)
      vehicle = image_semantic_1hot[self.vehicle_ids].sum(dim=0, keepdim=True)
      background = torch.ones(vehicle.shape)-vehicle-road


      # back to front
      input_image = torch.cat([background, road, vehicle], dim=0)   # 3xhxw

      return input_image


class VAE(nn.Module):
   def __init__(self):
      super(VAE, self).__init__()

      self.encoder = nn.Sequential(
            nn.Conv2d(3,32,3, 2),
            nn.ReLU(),
            nn.Conv2d(32,64,3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,128,3, 2),
            nn.ReLU(),
            nn.Conv2d(128,256,3, 2),
            nn.ReLU(),
            nn.Conv2d(256,512,3, 2),
            nn.ReLU(),
         )
      self.encoder_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.5),
         )

      self.fc_mu = nn.Linear(1024, 512)
      self.fc_var = nn.Linear(1024, 512)

      self.decoder_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2048),
            nn.Dropout(p=0.5),
         )
      
      self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2),
            nn.ReLU(),
            nn.Upsample(224)
         )


   def encode(self, x):
      x = self.encoder(x)
      x = torch.flatten(x, start_dim=1)
      x = self.encoder_fc(x)
      mu = self.fc_mu(x)
      logvar = self.fc_var(x)
      return mu, logvar

   def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return mu + eps*std

   def decode(self, z):
      x = self.decoder_fc(z)
      x = x.view(-1, 512, 2, 2)
      x = self.decoder(x)
      return x

   def forward(self, x):
      mu, logvar = self.encode(x)

      z = self.reparameterize(mu, logvar)

      return self.decode(z), mu, logvar

def visualize(preds, targets):

   wandb.log({"preds": [wandb.Image(transforms.ToPILImage()(preds[im]), caption="Cafe") for im in range(5)]})
   wandb.log({"targets": [wandb.Image(transforms.ToPILImage()(targets[im]), caption="Cafe") for im in range(5)]})


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_data, data, mu, logvar, kl_weight = 1e-3):

   criterion = nn.CrossEntropyLoss(weight = torch.tensor([.2, .2, .6]).to(device))
   
   data = torch.argmax(data, dim=1)

   recon_loss = criterion(recon_data, data)

   # https://arxiv.org/abs/1312.6114
   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   # KLD = 0

   return KLD*kl_weight + recon_loss


def train():
   dataset = Kitti360Semantic1Hot(args.data_folder, args.crop_size)

   model = VAE().to(device)

   optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


   for epoch in range(args.train_epochs):

      loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True)

      iterator = iter(loader)

      train_loss = 0

      for iteration in tqdm(range(len(loader))):

         this_batch = next(iterator)

         this_batch = this_batch.to(device)
      
         optimizer.zero_grad()
         recon_data, mu, logvar = model.forward(this_batch)
         loss = loss_function(recon_data, this_batch, mu, logvar, kl_weight=args.kl_weight)
         loss.backward()
         train_loss += loss.item()
         optimizer.step()

      print(f"epoch: {epoch}, loss: {train_loss}")
      wandb.log({'epoch': epoch, 'loss': train_loss})
      visualize(recon_data, this_batch)
   
      torch.save(model.state_dict(), f"{args.save_folder}/vae.pt")

if __name__ == "__main__":

   parser = argparse.ArgumentParser()


   parser.add_argument('--train_epochs', type=int, default=100)
   parser.add_argument('--learning_rate', type=float, default=1e-3)
   parser.add_argument('--kl_weight', type=float, default=1e-5)
   parser.add_argument('--crop_size', type=int, default=224)
   parser.add_argument('--batch_size', type=int, default=256)
   parser.add_argument('--num_workers', type=int, default=15)
   parser.add_argument('--save_folder', default="models/")
   parser.add_argument('--data_folder', default="/scratch/iamerich/kitti360/KITTI-360/data_2d_semantics/train/")
   
   args = parser.parse_args()

   device = torch.device("cuda:0")

   wandb.init(project="scene-rearrangement", name="new_vae")
   wandb.config.update(args) 

   print("args")
   print(args)

   train()