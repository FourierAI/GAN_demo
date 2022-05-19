import torch
import torch.nn as nn
import torch.nn.functional as F

class G_CGAN(nn.Module):

    def __init__(self, condition_size=10, latent_ratio=10, seq_size=50):
        super(G_CGAN, self).__init__()
        self.l1_in_size = condition_size
        self.latent_size = self.l1_in_size * latent_ratio
        self.l2_out_size = seq_size

        self.l1 = nn.Linear(self.l1_in_size, self.latent_size)
        self.l2 = nn.Linear(self.latent_size, self.l2_out_size)

    def forward(self, condition):
        x = self.l1(condition)
        x = F.relu(x)
        x = self.l2(x)
        return x

class D_CGAN(nn.Module):

    def __init__(self, seq_size=50, condition_size=10, latent_ratio=0.5):
        super(D_CGAN, self).__init__()
        self.total_size = seq_size + condition_size
        self.latent_size = int(self.total_size * latent_ratio)
        self.l1 = nn.Linear(self.total_size, self.latent_size)
        self.l2 = nn.Linear(self.latent_size, 1)

    def forward(self, sequence, condition):

        concatenate = torch.cat((sequence, condition), 1)
        x = self.l1(concatenate)
        x = F.relu(x)
        x = self.l2(x)
        x = F.sigmoid(x)
        return x
