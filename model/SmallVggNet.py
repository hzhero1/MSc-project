import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallVggNet(nn.Module):
    def __init__(self, num_classes=True):
        super(SmallVggNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3))
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))

        self.conv1 = nn.Sequential(  # input shape (3, input_size, input_size)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # output shape (16, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),  # output shape (16, 32, 32)
            nn.Dropout2d(0.2)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 32, 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # output shape (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # output shape (32, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 16, 16)
            nn.Dropout2d(0.3)
        )

        self.fc1 = nn.Sequential(nn.Linear(32 * 16 * 16, 1024),
                                 nn.Dropout2d(0.6))
        self.fc2 = nn.Linear(1024, num_classes)

        # self.conv3 = nn.Sequential(  # input shape (16, 32, 32)
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),  # output shape (64, 16, 16)
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),  # output shape (64, 16, 16)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(),
        #     nn.MaxPool2d(kernel_size=2)  # output shape (64, 8, 8)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output

