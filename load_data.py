import shutil
import numpy as np
import random
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


def train_val_split(datadir, valid_size=0.2, batch_size=64, input_size=64):
    train_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)
    val_data = datasets.ImageFolder(datadir,
                                    transform=val_transforms)
    test_data = datasets.ImageFolder(datadir,
                                     transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    print('train_idx len:', len(train_idx))
    print('test_idx len:', len(test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             sampler=val_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_data, sampler=val_sampler)

    return train_loader, val_loader, test_loader, train_data.class_to_idx


def train_val_split_augmentation_traditional(datadir, valid_size=0.2, batch_size=64, input_size=64):
    train_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                           torchvision.transforms.RandomHorizontalFlip(),
                                           torchvision.transforms.RandomRotation(20),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
    val_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)
    val_data = datasets.ImageFolder(datadir,
                                    transform=val_transforms)
    test_data = datasets.ImageFolder(datadir,
                                     transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    print('train_idx len:', len(train_idx))
    print('test_idx len:', len(test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             sampler=val_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_data, sampler=val_sampler)

    return train_loader, val_loader, test_loader, train_data.class_to_idx


def load_split_train_val(datadir_train, datadir_val, datadir_test, batch_size=64, input_size=64):
    train_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder(datadir_train, transform=train_transforms)
    val_data = datasets.ImageFolder(datadir_val, transform=val_transforms)
    test_data = datasets.ImageFolder(datadir_test, transform=test_transforms)

    print(train_data.classes)
    print(train_data.class_to_idx)
    print('number of train:', len(train_data))
    print('number of val:', len(val_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.class_to_idx
