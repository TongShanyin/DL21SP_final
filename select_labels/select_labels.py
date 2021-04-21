import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataloader_labels import CustomDataset
from contrastive import Encoder, train_transforms, validation_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', type=str)
args = parser.parse_args()

unsupset = CustomDataset(root='/dataset', split='unlabeled', transform=validation_transforms)
unsuploader = torch.utils.data.DataLoader(unsupset, batch_size=256, shuffle=False, num_workers=1)

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

#NUM_FEATURE = 2048
NUM_FEATURE = 512

encoder = Encoder()
classifier = LinearClassifier(NUM_FEATURE)
net = nn.Sequential(encoder, classifier)
net.load_state_dict(torch.load(args.model_checkpoint))
net = net.cuda()

print('Label selection')

print('use checkpoint:'+args.model_checkpoint)

net.eval()
for i, data in enumerate(unsuploader):
    inputs, labels, idx = data
    inputs, labels, idx = inputs.cuda(), labels.cuda(), idx.cuda()

    outputs = net(inputs)
    for index, i in enumerate(outputs):
        identifier = idx[index]
        sort, _ = torch.sort(i, descending=True)
        diff = sort[0] - sort[1]
        print('Id: %s, Diff: %s' % (identifier.item(), diff.item()), flush=True)

