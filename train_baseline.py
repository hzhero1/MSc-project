import shutil
import numpy
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from model.SmallVggNet import SmallVggNet
import matplotlib.pyplot as plt

data_dir_train = '../dataset/test_data/train'
data_dir_val = '../dataset/test_data/eval'
data_dir_test = '../dataset/test_data/test'
num_classes = 2


def load_split_train_val(datadir_train, datadir_val):
    train_transforms = transforms.Compose([transforms.Resize([32, 32]),
                                           transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Resize([32, 32]),
                                         transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize([32, 32]),
                                          transforms.ToTensor()])

    train_data = datasets.ImageFolder(datadir_train, transform=train_transforms)
    val_data = datasets.ImageFolder(datadir_val, transform=val_transforms)
    # test_data = datasets.ImageFolder(data_dir_test, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=64)
    val_loader = DataLoader(val_data, batch_size=64)
    # test_loader = DataLoader(test_data)

    return train_loader, val_loader


train_loader, val_loader = load_split_train_val(data_dir_train, data_dir_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallVggNet(num_classes).to(device)

num_epochs = 50
batch_size = 25
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

# torch.save(model.state_dict(), 'model.ckpt')

# test-the-model
# model.eval()  # it-disables-dropout
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in val_loader :
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Test Accuracy of the model: {} %'.format(100 * correct / total))
