import argparse
import os
import numpy as np
import math
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchvision.transforms as transforms
from torchvision.utils import save_image
from data import *
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from model import *
import matplotlib.pyplot as plt


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--samples", type=int, default=100_0000, help="sample size of the data")
parser.add_argument("--ht_dim", type=int, default=10, help="ht dimension")
parser.add_argument("--ecg_dim", type=int, default=200, help="Ecg dimension")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = Generator(3, opt.ecg_dim)
discriminator = Discriminator(opt.ecg_dim)


if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
dataset = EcgDataset(opt.samples, opt.ecg_dim)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (ecgs, hts) in enumerate(dataloader):

        # Configure input
        real_ecgs = Variable(ecgs.type(FloatTensor))
        hts = Variable(hts.type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input

        # Generate a batch of images
        fake_ecgs = generator(hts).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_ecgs)) + torch.mean(discriminator(fake_ecgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_ecgs = generator(hts)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_ecgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )
        if batches_done % opt.sample_interval == 0:
            plt.plot(fake_ecgs.cpu()[0].numpy()[0])
            plt.show()
            # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1

torch.save(generator, './G.pth')