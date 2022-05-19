import torch
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

G = torch.load('./G.pth')

BATCH_SIZE = 10
noise = torch.rand((BATCH_SIZE, 5))
condition = torch.ones(BATCH_SIZE).view(-1, 1)
paintings = G(noise, condition)
Y1 = paintings.data.numpy()[8]
Y2 = paintings.data.numpy()[1]
Y3 = paintings.data.numpy()[2]
plt.plot(Y1)
plt.plot(Y2)
plt.show()
