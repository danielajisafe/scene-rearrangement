import argparse
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np

from PIL import Image 
import cv2
from tqdm import tqdm
from glob import glob
import random
from os.path import join

import wandb

class Kitti360Semantic1Hot(Dataset):
   def __init__(self, data_dir, crop_size, train=True):
      train_folders = [
               "2013_05_28_drive_0000_sync", 
               "2013_05_28_drive_0007_sync", 
               "2013_05_28_drive_0010_sync",
               "2013_05_28_drive_0002_sync",
               "2013_05_28_drive_0004_sync",
               "2013_05_28_drive_0009_sync"
      ]

      val_folders = [
               "2013_05_28_drive_0003_sync", 
               "2013_05_28_drive_0006_sync"
      ]

      if(train):
         self.data = []
         for folder in train_folders:
            self.data.extend(glob(join(data_dir, folder, 'semantic', '*.png')))
      else:
         self.data = []
         for folder in val_folders:
            self.data.extend(sorted(glob(join(data_dir, folder, 'semantic', '*.png'))))


      self.crop_size = crop_size

      self.num_classes = 45

      self.road_ids = [7, 9]
      self.vehicle_ids = [26, 27, 28, 29, 30, 32, 33]

      self.data_transforms = transforms.Compose([
                     transforms.RandomCrop((350, 1300)),
                     transforms.RandomHorizontalFlip(),
                     transforms.Resize((self.crop_size, self.crop_size), interpolation=Image.NEAREST)
                     ])

   def __len__(self):
      return len(self.data)

   def __getitem__(self, index):

      image = cv2.imread(self.data[index])

      image = torch.Tensor(image)

      image = image.permute(2, 0, 1)

      image = self.data_transforms(image)
   
      image_semantic_id = image[0, :, :]

      ones = torch.ones(image_semantic_id.shape)
      zeros = torch.zeros(image_semantic_id.shape)

      image_semantic_1hot = torch.zeros(( self.num_classes, image.shape[1], image.shape[2]))	# shape = CxHxW
      mask_out = torch.zeros((image.shape[1], image.shape[2]))	# shape = HxW

      for i in self.road_ids+self.vehicle_ids:
         image_semantic_1hot[i] = torch.where(image_semantic_id == i, ones, zeros)

      # classes determined based on the labels provided by https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py

      road = image_semantic_1hot[self.road_ids].sum(dim=0, keepdim=True)
      vehicle = image_semantic_1hot[self.vehicle_ids].sum(dim=0, keepdim=True)
      background = torch.ones(vehicle.shape)-vehicle-road

      # back to front
      input_image = torch.cat([background, road, vehicle], dim=0)   # 3xhxw

      return {"image": input_image, "filename": self.data[index]}


class VAE(nn.Module):
   def __init__(self):
      super(VAE, self).__init__()

      self.relu = nn.ReLU()

      self.encoder_conv1 = nn.Conv2d(3,16,3, 2)
      self.encoder_conv2 = nn.Conv2d(16,32,3, 2)
      self.encoder_conv3 = nn.Conv2d(32,64,3, 2)
      self.encoder_conv4 = nn.Conv2d(64,128,3, 2)
      
      self.fc_mu = nn.Linear(2048, 512)
      self.fc_var = nn.Linear(2048, 512)

      self.decoder_fc = nn.Linear(512, 2048)

      self.decoder_conv1 = nn.ConvTranspose2d(128, 64, 3, 2)
      self.decoder_conv2 = nn.ConvTranspose2d(64, 32, 3, 2)
      self.decoder_conv3 = nn.ConvTranspose2d(32, 16, 3, 2)
      self.decoder_conv4 = nn.ConvTranspose2d(16, 3, 3, 2)
      
      self.softmax = nn.Softmax(1)


   def encode(self, x):
      x = self.encoder_conv1(x)
      x = self.relu(x)
      x = self.encoder_conv2(x)
      x = self.relu(x)
      x = self.encoder_conv3(x)
      x = self.relu(x)
      x = self.encoder_conv4(x)
      x = self.relu(x)

      x = torch.flatten(x, start_dim=1)
      mu = self.fc_mu(x)
      logvar = self.fc_var(x)
      return mu, logvar

   def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return mu + eps*std

   def decode(self, z):
      x = self.decoder_fc(z)
      x = x.view(-1, 128, 4, 4)
      x = self.decoder_conv1(x)
      x = self.relu(x)
      x = self.decoder_conv2(x)
      x = self.relu(x)
      x = self.decoder_conv3(x)
      x = self.relu(x)
      x = self.decoder_conv4(x)
      x = self.relu(x)
      x = self.softmax(x)

      return x

   def forward(self, x):
      mu, logvar = self.encode(x)

      z = self.reparameterize(mu, logvar)

      return self.decode(z), mu, logvar


def sample(model, std):

   z = torch.randn(5, 512).to(device)*std

   recon = model.decode(z)

   return recon


def visualize(preds, targets, sampled, loss, kld_loss, recon_loss, z, epoch):

   wandb.log({ "preds":       [wandb.Image(transforms.ToPILImage()(preds[im]), caption="Cafe",) for im in range(5)]}, step=epoch)
   wandb.log({ "targets":     [wandb.Image(transforms.ToPILImage()(targets[im]), caption="Cafe") for im in range(5)]}, step=epoch)
   wandb.log({ "Sampled":     [wandb.Image(transforms.ToPILImage()(sampled[im]), caption="Cafe",) for im in range(5)]}, step=epoch)
   wandb.log({ "loss": loss, "kld_loss": kld_loss, "recon_loss": recon_loss, "z_mean": torch.mean(z), "z_std": torch.std(z)}, step=epoch)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_data, data, mu, logvar, kl_weight = 1e-3):

   total_sum = torch.sum(data)

   invididual_weights = []

   for i in range(data.shape[1]):
      invididual_weights.append(total_sum/torch.sum(data[:, i]))

   criterion = nn.CrossEntropyLoss()

   criterion = nn.CrossEntropyLoss(weight = torch.tensor(invididual_weights).to(device))

   # criterion = nn.CrossEntropyLoss(weight = torch.tensor([2., 4., 8.]).to(device))

   data = torch.argmax(data, dim=1)

   recon_loss = criterion(recon_data, data)

   # https://arxiv.org/abs/1312.6114
   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   # KLD = 0

   return KLD*kl_weight + recon_loss, KLD, recon_loss


def train():
   dataset = Kitti360Semantic1Hot(args.data_folder, args.crop_size)

   model = VAE().to(device)

   optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


   for epoch in range(args.train_epochs):

      loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True)

      iterator = iter(loader)

      train_loss = 0
      kld_loss = 0
      total_recon_loss = 0

      for iteration in tqdm(range(len(loader))):

         this_batch = next(iterator)["image"]

         this_batch = this_batch.to(device)
      
         optimizer.zero_grad()
         recon_data, mu, logvar = model.forward(this_batch)
         loss, KLD, recon_loss = loss_function(recon_data, this_batch, mu, logvar, kl_weight=args.kl_weight)
         loss.backward()
         train_loss += loss.item()
         kld_loss += KLD.item()
         total_recon_loss += recon_loss.item()
         optimizer.step()

      sampled = sample(model, torch.std(mu))
      visualize(recon_data, this_batch, sampled, train_loss, kld_loss, total_recon_loss, mu, epoch)
      print(f"epoch: {epoch}, loss: {train_loss}")
   
      torch.save(model.state_dict(), f"{args.save_folder}/vae.pt")

def write_images(this_batch, folder, image_index, dest):

   this_batch = transforms.Resize((256, 256), interpolation=Image.NEAREST)(this_batch).cpu().numpy()

   out_image = np.zeros((len(this_batch), 256, 256))

   # creating the index mask needed for loss calculation
   for i in range(this_batch.shape[1]):
      out_image += i * this_batch[:, i]


   for i in range(len(this_batch)):

      cv2.imwrite(f"{dest}/{folder}/img/{image_index:010d}.png", out_image[i])
      cv2.imwrite(f"{dest}/{folder}/label/{image_index:010d}.png", out_image[i])

      image_index += 1
   
   return image_index

def test():

   import cv2
   from shutil import copyfile

   model = VAE().to(device)
   model.load_state_dict(torch.load(f"{args.save_folder}/vae.pt"))
   model.eval()

   for param in model.parameters():
        param.requires_grad = False

   dataset = Kitti360Semantic1Hot(args.data_folder, args.crop_size, train=False)
   loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers=0, shuffle=False)
   iterator = iter(loader)

   dest = "/scratch/iamerich/SPADE/datasets/kitti360/validation"

   gt_image_index = 0
   recon_image_index = 0
   opt_image_index = 0


   for iteration in tqdm(range(len(iterator))):
      batch_dict = next(iterator)
      this_batch = batch_dict["image"].to(device)

      file_name = batch_dict["filename"]

      gt_image_index = write_images(this_batch, "ground_truth", gt_image_index, dest)

      mu, _ = model.encode(this_batch)
      initial_estimate = model.decode(mu)

      recon_image_index = write_images(initial_estimate, "reconstructed", recon_image_index, dest)

      mse_loss = nn.MSELoss()

      mu.requires_grad = True

      next_image = [this_batch[(i+1)%len(this_batch)] for i in range(len(this_batch))]

      next_image = torch.stack(next_image)

      optimizer = optim.Adam([mu], lr=args.optimization_rate)

      for opt_iteration in tqdm(range(args.optimization_iterations)):

         optimizer.zero_grad()

         x_hat = model.decode(mu)

         # loss = mse_loss(batch, recon_data)
         loss = mse_loss(next_image[:, 2], x_hat[:, 2])

         loss.backward()
         optimizer.step()
      
      final_image = model.decode(mu)

      opt_image_index = write_images(final_image.detach(), "optimized", opt_image_index, dest)


   

if __name__ == "__main__":

   parser = argparse.ArgumentParser()


   parser.add_argument('--train_epochs', type=int, default=100)
   parser.add_argument('--optimization_iterations', type=int, default=10000)
   parser.add_argument('--learning_rate', type=float, default=1e-3)
   parser.add_argument('--optimization_rate', type=float, default=1e-3)
   parser.add_argument('--kl_weight', type=float, default=1e-7)
   parser.add_argument('--crop_size', type=int, default=79)
   parser.add_argument('--batch_size', type=int, default=2048)
   parser.add_argument('--num_workers', type=int, default=15)
   parser.add_argument('--save_folder', default="models/")
   parser.add_argument('--data_folder', default="/scratch/iamerich/kitti360/KITTI-360/data_2d_semantics/train/")
   
   args = parser.parse_args()

   device = torch.device("cuda:1")

   wandb.init(project="scene-rearrangement", name="optimizing_input")
   wandb.config.update(args) 

   print("args")
   print(args)

   # train()
   test()
