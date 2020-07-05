import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from model.SmallVggNet import SmallVggNet
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

data_dir_train = '../dataset/test_data/train'
data_dir_val = '../dataset/test_data/eval'
data_dir_test = '../dataset/test_data/test'
num_classes = 5
input_size = 32
batch_size = 64


def train_val_split(datadir, valid_size=0.2, batch_size=64):
    train_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                           transforms.ToTensor()])

    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    print('train_idx len:', len(train_idx))
    print('test_idx len:', len(test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(train_data,
                                             sampler=test_sampler, batch_size=batch_size)

    return trainloader, testloader


train, test = train_val_split(data_dir_train, valid_size=.2, batch_size=batch_size)
