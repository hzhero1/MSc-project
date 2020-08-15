import argparse
import os
import random
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# t = torch.tensor([[1, 2, 3], [2, 3, 4], [4, 4, 5]])
# t = t.float()
# # print(t.diagonal().sum())
# # print(t.sum())
# print(t)
# print(t.diagonal())
# # print(t.diagonal() / t.sum(0))
# # print(t.diagonal() / t.sum(1))
# print(t.sum(0))
# print(t.sum(1))
# # print(t.sum(0))
# # print(t.sum(1))
# print(t.sum(0) + t.sum(1))
# print(t.sum(0) * t.sum(1))
# print((torch.div(t.sum(0) + t.sum(1), (t.sum(0) * t.sum(1)))))
# # print('\n-----------------------\nEvaluation on test data\n-----------------------')
# # print('{:20}{}'.format('1', 2))
# # print()

# y_ = (torch.rand(5, 1) * 2).type(torch.LongTensor).squeeze()
# print(y_)

# Create the dataset
dataset = dset.ImageFolder(root="../cdcgan_samples/gray_sample",
                           transform=transforms.Compose([
                               # transforms.Resize((128, 128)),
                               transforms.ToTensor(),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=9,
                                         shuffle=True)
#
# # Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(0)[:9], nrow=3, padding=2).cpu(), (1, 2, 0)))
plt.show()
