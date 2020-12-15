# scene-rearrangement

This is the github repository for the project Novel Scene Generation Via Decomposition.

The framework is based on Pytorch library.

4 models are developed here:
1. vae_test (Vanilla VAE)
2. vae_multi_stage (Stacked VAE)
3. vae-gan-w_3-2-5 ( Markov VAE-WGAN)
4. shift_gan-h10 (Shift-WGAN)


Requirements are:
- pytorch
- open cv
- torchsummary
- tqdm

Folder structures are as follows:
- 

- main.py:  starting point. 
- optimization.py: guided test-time optimization for rearrenging objects in Vanilla VAE
- config/*.yml:  contains the configuration of each model
- data/: contains the pytorch dataset classes
- losses/: contains the loss functions
- model/: contains the pytorch nn.Module classes for each model architecture
- test/: contains notebooks for testing purposes (currently holds the dataset visualization notebook)
- trainer/: contains trainer classes which handles training and testing of models (basically connects datasets, model architectures, visualizations, etc)
- utils/: contains utility scripts such as: logging, and FID calculator
- visualization/: contains wandb related visualization functions


FID score calculation:
- 
We used the method in here to do so: https://github.com/mseitzer/pytorch-fid

to run it we just do:
`python -m pytorch_fid path/to/dataset1 path/to/dataset2`
 
Model Training Runs on  "_Weights & Biasses_" website 
- 
1. Vanilla VAE:  https://wandb.ai/hooman007/scene-rearrangement/runs/1idedws0?workspace=user-hooman007

2. Stacked VAE: https://wandb.ai/iamerichedlin/scene-rearrangement/runs/2jj1kr7s?workspace=user-iamerichedlin

3. Markov VAE-WGAN: https://wandb.ai/dnamrata/scene-rearrangement/runs/1775r3b7?workspace=user-dnamrata

4. Shift-WGAN: https://wandb.ai/dnamrata/scene-rearrangement/runs/fh58w43g?workspace=user-dnamrata
