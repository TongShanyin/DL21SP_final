# Training process for SimCLR

import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from simclr_loader import ContrastiveDataset
from contrastive import SimCLR, NTXent, train_transforms, validation_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
#parser.add_argument('--checkpoint_net', type=str) # uncomment if use the previous checkpoints
args = parser.parse_args()


BATCH_SIZE = 1024

trainset = ContrastiveDataset(root='/dataset', split="unlabeled", transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


NUM_FEATURE = 2048 #renet50

NUM_LATENT = 128

TEMPERATURE = 0.1


net = SimCLR(NUM_FEATURE, NUM_LATENT).cuda()

#net.load_state_dict(torch.load(args.checkpoint_net)) # uncomment if use previous weights
#print('use checkpoint:'+args.checkpoint_net)

criterion = NTXent(BATCH_SIZE, TEMPERATURE).cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140, 160, 180], gamma=0.5)


os.makedirs(args.checkpoint_dir, exist_ok=True)

print('Start Training')
tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(100):
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
    # save checkpoints per epoch to avoid the early ending of training
    checkpoint_name = "simclr_"
    torch.save(net.encoder.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name+"encoder.pth"))
    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name+"net.pth"))
    

    scheduler.step()



print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))
