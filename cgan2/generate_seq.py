import torch
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

G = torch.load('./G.pth')

BATCH_SIZE = 10

# condition_linear = torch.ones(BATCH_SIZE).view(-1, 1)
# linear = G(condition_linear)
# Y = linear.data.numpy()[8]
# plt.plot(Y)
# plt.show()

condition_quadratic = torch.ones(BATCH_SIZE)*2
condition_quadratic = condition_quadratic.view(-1, 1)
quadratic = G(condition_quadratic)
Y = quadratic.data.numpy()[1]
plt.plot(Y)
plt.show()