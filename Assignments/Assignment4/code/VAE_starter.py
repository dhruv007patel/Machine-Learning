#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import imageio
import torch.nn.functional as funcs
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from sklearn.manifold import TSNE
import seaborn as sns


def scatter_plot(latent_representations, labels):

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_image_plot = tsne.fit_transform(latent_representations)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        tsne_image_plot[:, 0],
        tsne_image_plot[:, 1],
        hue=labels,
        data=latent_representations,
        palette=sns.color_palette('husl', 10),
        legend='full',
        alpha=0.5,
        )


def Plot_Kernel(_model):
    model_weight = _model.encoder.encoder[0].weight
    return model_weight


def display_images_in_a_row(images, file_path='./tmp.png',
                            display=True):
    save_image(images.view(-1, 1, 28, 28), '{}'.format(file_path))
    if display is True:
        plt.imshow(mpimg.imread('{}'.format(file_path)))


# Defining Model

class VAE_Trainer(object):

    def __init__(
        self,
        autoencoder_model,
        learning_rate=1e-3,
        path_prefix='',
        ):
        self.device = torch.device(('cuda'
                                    if torch.cuda.is_available() else 'cpu'
                                   ))
        self.init_dataset(path_prefix)
        self.model = autoencoder_model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                lr=learning_rate, weight_decay=1e-5)

    def init_dataset(self, path_prefix=''):
        transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))])
        trainTransform = \
            torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307, ), (0.3081,
                ))])
        trainset = \
            torchvision.datasets.FashionMNIST(root='{}/./data'.format(path_prefix),
                train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset,
                batch_size=32, shuffle=False, num_workers=4)
        valset = \
            torchvision.datasets.FashionMNIST(root='{}/./data'.format(path_prefix),
                train=False, download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=32,
                shuffle=False, num_workers=4)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.valset = valset
        self.trainset = trainset

    # Readme

    def loss_function(
        self,
        recon_x,
        x,
        mu,
        logvar,
        ):
        BCE = funcs.mse_loss(recon_x, x.view(-1, 784))
        KLD = -0.5 * torch.sum(logvar.exp() - logvar - 1 + mu ** 2)
        Loss = BCE + KLD  # Case 1: with KLD

        # Loss = BCE     # Case 2: without KLD

        return Loss

    def get_train_set(self):
        images = torch.vstack([x for (x, _) in self.train_loader])  # get the entire train set
        return images

    def get_val_set(self):
        images = torch.vstack([x for (x, _) in self.val_loader])  # get the entire val set
        return images

    def train(self, epoch):

        # Note that you need to modify both trainer and loss_function for the VAE model

        self.model.train()
        train_loss = 0
        for (batch_idx, (data, _)) in \
            tqdm(enumerate(self.train_loader),
                 total=len(self.train_loader)):
            data = data.reshape(-1, 784)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            (recon_batch, mu, log_var) = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_loss /= len(self.train_loader.dataset) / 32  # 32 is the batch size
        print '====> Epoch: {} Average loss: {:.4f}'.format(epoch,
                train_loss)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for (i, (data, _)) in tqdm(enumerate(self.val_loader),
                    total=len(self.val_loader)):
                data = data.reshape(-1, 784)
                data = data.to(self.device)
                (recon_batch, mu, logvar) = self.model(data)
                val_loss += self.loss_function(recon_batch, data, mu,
                        logvar).item()

        val_loss /= len(self.val_loader.dataset) / 32  # 32 is the batch size
        print '====> Val set loss (reconstruction error) : {:.4f}'.format(val_loss)
