import torch
import torch.nn as nn


class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.AvgPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(16*4*4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = self.pool(x)
    x = torch.relu(self.conv2(x))
    x = self.pool(x)
    x = self.flatten(x)
    x =  torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x