import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from torchvision import datasets, transforms, models

#from dataloader import CustomDataset
from simclr_loader import ContrastiveDataset
from contrastive import SimCLR, NTXent, train_transforms, validation_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

BATCH_SIZE = 256

trainset = ContrastiveDataset(root='/dataset', split="unlabeled", transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#evalset = ContrastiveDataset(root='/dataset', split="val", transform=validation_transforms)
#evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

NUM_FEATURE = 2048
NUM_LATENT = 128
TEMPERATURE = 1.

net = SimCLR(NUM_FEATURE, NUM_LATENT).cuda()
criterion = NTXent(BATCH_SIZE, TEMPERATURE).cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=0.3, weight_decay=1e-6)

os.makedirs(args.checkpoint_dir, exist_ok=True)

print('Start Training')
tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(50):
    net.train()
    

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [img, img]
        xi, xj = data
        xi, xj = xi.cuda(), xj.cuda()

        zi, zj = net(xi,xj)
        loss = criterion(zi, zj)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100), flush=True)
            running_loss = 0.0
    
    tac = time.perf_counter()
    print("Time elapsed: " + str(tac - tic))
    if epoch % 5 == 4:
        torch.save(net.encoder.state_dict(), os.path.join(args.checkpoint_dir, "simclr_encoder_ep_%s" % str(epoch+1)))
        print("Saved intermediate checkpoint to encoder_ep_%s" % str(epoch))
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "simclr_ep_%s" % str(epoch+1)))
        print("Saved intermediate checkpoint to simclr_ep_%s" % str(epoch))

#    net.eval()
#    running_eval_loss = 0.0
#    for i, data in enumerate(evalloader):
#        xi, xj = data
#        xi, xj = xi.cuda(), xj.cuda()
#        zi, zj = net(xi,xj)
#        loss = criterion(zi, zj)
#        running_eval_loss += loss.item()
#        if i % 10 == 9:
#            print('Eval loss:' + str(running_eval_loss / 10))
#            running_eval_loss = 0.0

print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))
