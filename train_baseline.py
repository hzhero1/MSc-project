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

manualSeed = 1

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

data_dir_train = '../dataset/The_CNBC_Face_Database'
data_dir_val = '../dataset/test_data/eval'
data_dir_test = '../dataset/test_data/test'
num_classes = 5
input_size = 32
batch_size = 128


def train_val_split(datadir, valid_size=0.2, batch_size=64):
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


def train_val_split_augmentation_traditional(datadir, valid_size=0.2, batch_size=64):
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


def load_split_train_val(datadir_train, datadir_val, batch_size=64):
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
    test_data = datasets.ImageFolder(data_dir_test, transform=test_transforms)

    print(train_data.classes)
    print(train_data.class_to_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data)

    return train_loader, val_loader, test_loader, train_data.class_to_idx


# train_loader, val_loader, test_loader, labels_idx = load_split_train_val(data_dir_train, data_dir_val)
train_loader, val_loader, test_loader, labels_idx = train_val_split_augmentation_traditional(data_dir_train, 0.2, 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallVggNet(num_classes).to(device)

num_epochs = 50
learning_rate = 0.001
train_losses, val_losses = [], []

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()

    correct = 0
    total = 0
    for data, labels in val_loader:
        data = data.to(device)
        labels = labels.to(device)

        output = model(data)
        loss = criterion(output, labels)

        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        val_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'
          .format(epoch, train_loss, val_loss, 100 * correct / total))

torch.save(model.state_dict(), 'model.ckpt')

# test-the-model
model.eval()  # it-disables-dropout
confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
print('\nclasses:', labels_idx)
print(confusion_matrix.diag() / confusion_matrix.sum(1))
print('\nTest Accuracy of the model: {} %'.format(100 * correct / total))
