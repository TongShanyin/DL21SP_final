# Feel free to modifiy this file.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

team_id = 6
team_name = "SMC"
email_address = "ccp5804@nyu.edu"

resize_size=96
validation_transforms = transforms.Compose([
                                    transforms.Resize(size=resize_size),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                   # normalize,
                                ])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Encoder(nn.Module): # resnet50 after average pooling
    def __init__(self):
        super(Encoder, self).__init__()
       # self.encoder = torchvision.models.resnet50()
        self.encoder = torchvision.models.resnet18()
        self.encoder.fc = Identity()

    def forward(self, x):
        x = self.encoder(x)
        return x # resnet50: 2048, resnet 18: 512

class LinearClassifier(torch.nn.Module):

  def __init__(self,num_feature):
    super().__init__()
    self.linear1 = nn.Linear(num_feature, 2048)
    self.linear2 = nn.Linear(2048, 1024)
    self.linear3 = nn.Linear(1024, 800)

    self.dropout1 = torch.nn.Dropout(p=0.5)
    self.dropout2 = torch.nn.Dropout(p=0.5)
    self.bn1 = torch.nn.BatchNorm1d(num_feature)
    self.bn2 = torch.nn.BatchNorm1d(2048)
    self.bn3 = torch.nn.BatchNorm1d(1024)

  def forward(self, x):
    x = torch.reshape(x, (x.size(0), -1))
    x = self.bn1(x)
    x = self.dropout1(x)
    x = F.relu(self.linear1(x))
    x = self.bn2(x)
    x = self.dropout2(x)
    x = F.relu(self.linear2(x))
    x = self.bn3(x)
    x = self.linear3(x)
    return x

NUM_FEATURE = 512


def get_model():
    encoder = Encoder()
    classifier = LinearClassifier(NUM_FEATURE)
    net = nn.Sequential(encoder, classifier)
    return net

eval_transform = validation_transforms
