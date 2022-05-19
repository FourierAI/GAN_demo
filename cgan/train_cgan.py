import torch
import torch.nn as nn
import numpy as np
from model import *
from data import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log')

torch.manual_seed(1)    # reproducible
np.random.seed(1)

EPOCH = 50
# 超参数
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator

# noise size
noise_size = 5
condition_size = 1
latent_ratio = 10
seq_size = 50

G = G_CGAN(noise_size, condition_size, latent_ratio, seq_size)

latent_ratio = 0.5
D = D_CGAN(seq_size, condition_size, latent_ratio)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

if __name__ == '__main__':

    # todo address size match problem
    #  solved
    data = PaintingDataset()
    dataloader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)

    GLOSS = []
    DLOSS = []
    for i in range(EPOCH):
        step = 0
        for paintings, condition in dataloader:
            noise = torch.randn(BATCH_SIZE, noise_size, requires_grad=True)
            G_paintings = G(noise, condition)
            prob_artist1 = D(G_paintings, condition)
            G_loss = torch.mean(torch.log(1. - prob_artist1))
            GLOSS.append(float(G_loss.data))
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            prob_artist0 = D(paintings, condition)  # D try to increase this prob
            prob_artist1 = D(G_paintings.detach(), condition)  # D try to reduce this prob
            D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
            DLOSS.append(float(D_loss.data))
            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)  # reusing computational graph
            opt_D.step()
            if step % 100 == 0:
                print(f'G_loss mean:{G_loss}, D_loss mean:{D_loss}')
                writer.add_scalar('G loss', G_loss, step)
                writer.add_scalar('D loss', D_loss, step)
            step += 1

    torch.save(G, './G.pth')
    torch.save(D, './D.pth')
