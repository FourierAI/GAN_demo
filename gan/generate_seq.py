import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
G = torch.load('./G.pth')

G_ideas = torch.rand(5)
seq = G(G_ideas)
X = np.arange(1, 16, 1)
Y = seq.data.numpy()
plt.plot(seq.data.numpy())
plt.show()