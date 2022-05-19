import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(1)    # reproducible
np.random.seed(1)

# 超参数
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# 真实的画家画
def artist_works():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    # f(x) = ax*2+a-1
    # a uniform \in [1,2]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


G = nn.Sequential(                      # Generator
    nn.Linear(N_IDEAS, 128),            # random ideas (could from normal distribution)
    nn.LeakyReLU(0.01),
    nn.Linear(128, ART_COMPONENTS),     # making a painting from these random ideas
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(ART_COMPONENTS, 128),     # receive art work either from the famous artist or a newbie like G
    nn.LeakyReLU(0.01),
    nn.Linear(128, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

if __name__ == '__main__':

    DLOSS = []
    GLOSS = []
    for step in range(10000):
        artist_paintings = artist_works()  # real painting from artist
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)  # random ideas\n
        G_paintings = G(G_ideas)  # fake painting from G (random ideas)
        prob_artist1 = D(G_paintings)  # D try to reduce this prob
        G_loss = torch.mean(torch.log(1. - prob_artist1))
        GLOSS.append(float(G_loss.data))
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        prob_artist0 = D(artist_paintings)  # D try to increase this prob
        prob_artist1 = D(G_paintings.detach())  # D try to reduce this prob
        D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        DLOSS.append(float(D_loss.data))
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)  # reusing computational graph
        opt_D.step()
        if step % 100 == 0:
            print(f'G_loss mean:{G_loss}, D_loss mean:{D_loss}')
            writer.add_scalar('G loss', G_loss, step)
            writer.add_scalar('D loss', D_loss, step)

    torch.save(G, './G.pth')
    torch.save(D, './D.pth')

    plt.plot(DLOSS, label = 'D loss')
    plt.plot(GLOSS, label = 'G loss')

    plt.show()
    writer.close()