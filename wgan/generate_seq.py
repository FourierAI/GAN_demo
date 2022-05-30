import torch
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

G = torch.load('./G.pth')

if cuda:
    G = G.cuda()


BATCH_SIZE = 10

# condition_linear = torch.ones(BATCH_SIZE).view(-1, 1)
# linear = G(condition_linear)
# Y = linear.data.numpy()[8]
# plt.plot(Y)
# plt.show()

condition_quadratic = torch.ones(BATCH_SIZE)*0

condition_quadratic = condition_quadratic.view(-1, 1)
if cuda:
    condition_quadratic = condition_quadratic.type(LongTensor)

quadratic = G(condition_quadratic)
Y = quadratic.data.cpu().numpy()[1][0]
plt.plot(Y)
plt.show()

