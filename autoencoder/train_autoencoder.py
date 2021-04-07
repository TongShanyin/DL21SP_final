# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 

import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from autoencoder import Encoder, Decoder, train_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

trainset = CustomDataset(root='/dataset', split="unlabeled", transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

encoder = Encoder()
decoder = Decoder()
net = nn.Sequential(encoder, decoder).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

print('Start Training')
tic = time.perf_counter()

net.train()
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))

os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "encoder_big_10ep.pth"))

print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'encoder_big_10ep.pth')}")
