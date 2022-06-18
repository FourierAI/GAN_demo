import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, bcg_dim, ecg_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(bcg_dim, 128, normalize=False),
            *block(128, 64),
            *block(64, 32),
            *block(32, 64),
            nn.Linear(64, ecg_dim),
            nn.Tanh()
        )

    def forward(self, bcg):
        ecg = self.model(bcg)
        return ecg


class Discriminator(nn.Module):
    def __init__(self, ecg_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(ecg_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, ecg):
        validity = self.model(ecg)
        return validity