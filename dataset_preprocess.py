import shutil
import numpy
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

data_dir_train = '../dataset/Caltech101/train'
data_dir_val = '../dataset/Caltech101/train'


def load_split_train_val(datadir_train, datadir_val):
    train_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                           transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                         transforms.ToTensor()])

    train_data = datasets.ImageFolder(datadir_train, transform=train_transforms)
    val_data = datasets.ImageFolder(datadir_val, transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)

    return train_loader, val_loader


train_loader, val_loader = load_split_train_val(data_dir_train, data_dir_val)
