# Implementation of classifier on top of autoencoder features

import torch
from torch import nn
from torch.nn import functional as F

class Classifier(torch.nn.Module):

  LINEAR_SIZE = 27 * 16 * 16

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.conv2 = torch.nn.Conv2d(in_channels=9, out_channels=27, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.linear1 = nn.Linear(self.LINEAR_SIZE, 800)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = torch.reshape(x, (x.size(0), -1))
    x = F.relu(self.linear1(x))
    return x


# For use with tightrope autoencoder
class LinearClassifier(torch.nn.Module):

  LINEAR_SIZE = 96 * 4 * 4

  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(self.LINEAR_SIZE, self.LINEAR_SIZE)
    self.linear2 = nn.Linear(self.LINEAR_SIZE, self.LINEAR_SIZE)
    self.linear3 = nn.Linear(self.LINEAR_SIZE, self.LINEAR_SIZE)
    self.linear4 = nn.Linear(self.LINEAR_SIZE, 800)

  def forward(self, x):
    x = torch.reshape(x, (x.size(0), -1))
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = self.linear4(x)
    return x

