# Implementation of classifier on top of autoencoder features

import torch
from torch import nn

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
