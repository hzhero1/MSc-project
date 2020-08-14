import shutil
import numpy as np
import random
import torch
from torch import nn
from torch import optim
from model.SmallVggNet import SmallVggNet
import matplotlib.pyplot as plt
import pickle
from data_loader import train_val_split, train_val_split_augmentation_traditional, load_split_train_val, \
    load_split_train_val_aug_traditional

manualSeed = 1

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model_name = 'model_weights/baseline.ckpt'
# data_dir = '../dataset/The_CNBC_Face_Database'

# data_dir_train = '../dataset/CNBC_4_classes/The_CNBC_Face_Database_aug_dcgan/train'
# data_dir_val = '../dataset/CNBC_4_classes/The_CNBC_Face_Database_aug_dcgan/val'
# data_dir_test = '../dataset/CNBC_4_classes/The_CNBC_Face_Database_aug_dcgan/test'

# data_dir_train = '../dataset/test_data/train'
# data_dir_val = '../dataset/test_data/eval'
# data_dir_test = '../dataset/test_data/test'

attribute_name = 'hair_color_stargan_aug'

# CelebA
data_dir_train = "../dataset/celeba/celeba_subset/" + attribute_name + "/train/"
data_dir_val = "../dataset/celeba/celeba_subset/" + attribute_name + "/val/"
data_dir_test = "../dataset/celeba/celeba_subset/" + attribute_name + "/test/"

# Losses directory
loss_train_dir = "loss/64_loss_train_celeba_stargan.txt"
loss_val_dir = "loss/64_loss_val_celeba_stargan.txt"
acc_train_dir = "accuracy/64_acc_train_celeba_stargan.txt"
acc_val_dir = "accuracy/64_acc_val_celeba_stargan.txt"

num_classes = 3
input_size = 64
batch_size = 128

train_loader, val_loader, test_loader, labels_idx = load_split_train_val(data_dir_train, data_dir_val, data_dir_test,
                                                                         batch_size, input_size)

# train_loader, val_loader, test_loader, labels_idx = train_val_split(data_dir, 0.2, batch_size, input_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallVggNet(num_classes).to(device)

num_epochs = 20
learning_rate = 0.001
train_losses, val_losses = [], []
train_accs, val_accs = [], []

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    val_loss = 0.0

    model.train()

    correct_train = 0
    total_train = 0
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    model.eval()

    correct_val = 0
    total_val = 0
    for data, labels in val_loader:
        data = data.to(device)
        labels = labels.to(device)

        output = model(data)
        loss = criterion(output, labels)

        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

        val_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    train_acc = correct_train / total_train
    val_acc = correct_val / total_val
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        'Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'
            .format(epoch, train_loss, 100 * train_acc, val_loss, 100 * val_acc))

torch.save(model.state_dict(), model_name)

with open(loss_train_dir, "wb") as f:
    pickle.dump(train_losses, f)
with open(loss_val_dir, "wb") as f:
    pickle.dump(val_losses, f)
with open(acc_train_dir, "wb") as f:
    pickle.dump(train_accs, f)
with open(acc_val_dir, "wb") as f:
    pickle.dump(val_accs, f)

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
confusion_matrix = confusion_matrix.float()

print('\n-----------------------\nEvaluation on test data\n-----------------------')
print('Confusion matrix:\n', confusion_matrix)
print('\nPer class evaluation: ')
print('{:15}{}'.format('Classes', labels_idx))
precision = torch.div(confusion_matrix.diag(), confusion_matrix.sum(0))
recall = torch.div(confusion_matrix.diagonal(), confusion_matrix.sum(1))
print('{:15}{}'.format('Precision', precision))
print('{:15}{}'.format('Recall', recall))
print('{:15}{}'.format('F1 Score', torch.div(2 * precision * recall, precision + recall)))
# print('Accuracy: ', torch.true_divide(confusion_matrix.diagonal().sum(), confusion_matrix.sum()))
print('\nTest Accuracy of the model: {} %'.format(100 * correct / total))

# print('\n-----------------------\nEvaluation on test data\n-----------------------')
# print('Confusion matrix:\n', confusion_matrix)
# print('\nPer class evaluation: ')
# print('{:15}{}'.format('Classes', labels_idx))
# precision = torch.true_divide(confusion_matrix.diag(), confusion_matrix.sum(0))
# recall = torch.true_divide(confusion_matrix.diagonal(), confusion_matrix.sum(1))
# print('{:15}{}'.format('Precision', precision))
# print('{:15}{}'.format('Recall', recall))
# print('{:15}{}'.format('F1 Score', torch.true_divide(2 * precision * recall, precision + recall)))
# # print('Accuracy: ', torch.true_divide(confusion_matrix.diagonal().sum(), confusion_matrix.sum()))
# print('\nTest Accuracy of the model: {} %'.format(100 * correct / total))
