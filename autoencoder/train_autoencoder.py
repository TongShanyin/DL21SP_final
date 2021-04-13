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
from autoencoder_resnet import Encoder, Decoder, train_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

trainset = CustomDataset(root='/dataset', split="unlabeled", transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

evalset = CustomDataset(root='/dataset', split="val", transform=train_transforms)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

encoder = Encoder()
decoder = Decoder()
net = nn.Sequential(encoder, decoder)
#net = torch.nn.DataParallel(net)
net = net.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

os.makedirs(args.checkpoint_dir, exist_ok=True)

print('Start Training')
tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(30):
    net.train()

    torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "baselineresnet_encoder_ep_%s" % str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(args.checkpoint_dir, "baselineresnet_decoder_ep_%s" % str(epoch)))
    print("Saved intermediate checkpoint to baselineresnet_encoder_ep_%s" % str(epoch))
    tac = time.perf_counter()
    print("Time elapsed: " + str(tac - tic))

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
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10), flush=True)
            running_loss = 0.0
    
    net.eval()
    running_eval_loss = 0.0
    for i, data in enumerate(evalloader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        running_eval_loss += loss.item()
        if i % 10 == 9:
            print('Eval loss:' + str(running_eval_loss / 10))
            running_eval_loss = 0.0

print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))

