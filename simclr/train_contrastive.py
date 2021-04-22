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
parser.add_argument('--checkpoint_net', type=str)
args = parser.parse_args()

BATCH_SIZE = 256
#BATCH_SIZE = 512
#BATCH_SIZE = 1024
#BATCH_SIZE = 2048

trainset = ContrastiveDataset(root='/dataset', split="unlabeled", transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#evalset = ContrastiveDataset(root='/dataset', split="val", transform=validation_transforms)
#evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

#NUM_FEATURE = 2048 #renet50
NUM_FEATURE = 512 #resnet18
NUM_LATENT = 128
#TEMPERATURE = 1.
#TEMPERATURE = 0.5
#TEMPERATURE = 0.3
#TEMPERATURE = 0.1
TEMPERATURE = 0.05

net = SimCLR(NUM_FEATURE, NUM_LATENT).cuda()

net.load_state_dict(torch.load(args.checkpoint_net)) # use previous weights
print('use checkpoint:'+args.checkpoint_net)

criterion = NTXent(BATCH_SIZE, TEMPERATURE).cuda()

#optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)

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
    checkpoint_name = "simclr_resnet18_norm_256t005sgd2_"
    torch.save(net.encoder.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name+"encoder"))
    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name+"net"))
    if epoch % 10 == 9:
        #checkpoint_name = "simclr_resnet18_2048lr4_"
        torch.save(net.encoder.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name+f"encoder_ep_{epoch+1}"))
        print("Saved intermediate checkpoint to:"+checkpoint_name+f"encoder_ep_{epoch+1}")
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name+f"ep_{epoch+1}"))
        print("Saved intermediate checkpoint to:"+checkpoint_name+f"ep_{epoch+1}")

    scheduler.step()

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
