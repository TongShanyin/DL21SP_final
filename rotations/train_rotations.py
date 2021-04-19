# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 

import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from rotation_dataset import RotationDataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    transforms.RandomRotation(25),
                                    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

trainset = RotationDataset(root='/dataset', split="unlabeled", transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=192, shuffle=True, num_workers=2)

evalset = RotationDataset(root='/dataset', split="val", transform=train_transforms)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=192, shuffle=False, num_workers=2)

net = torchvision.models.alexnet(pretrained=False)
net.classifier[6] = torch.nn.Linear(4096, 4)
net.load_state_dict(torch.load('checkpoints/rotations_aug/rotations_ep_15'))
net = net.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)

os.makedirs(args.checkpoint_dir, exist_ok=True)

def compute_accuracy(outputs, rotation):
  predictions = torch.argmax(outputs, dim=1)
  correct = (predictions == rotation)
  return sum(correct).item() / list(correct.size())[0]

print('Start Training')
tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(100):
    net.train()

    if (epoch % 5 == 0) and (epoch != 0):
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "rotations_ep_%s" % str(epoch)))
        print("Saved intermediate checkpoint to rotations_ep_%s" % str(epoch))
        tac = time.perf_counter()
        print("Time elapsed: " + str(tac - tic))

    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(trainloader):
        inputs, rotation = data
        inputs, rotation = inputs.cuda(), rotation.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, rotation)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        accuracy = compute_accuracy(outputs, rotation)
        running_accuracy += accuracy
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10), flush=True)
            print('[%d, %5d] ------- ACC: %.3f' % (epoch + 1, i + 1, running_accuracy / 10), flush=True)
            running_loss = 0.0
            running_accuracy =0.0
    
    scheduler.step()
    net.eval()
    running_eval_loss = 0.0
    running_eval_accuracy = 0.0
    for i, data in enumerate(evalloader):
        inputs, rotation = data
        inputs, rotation = inputs.cuda(), rotation.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, rotation)
        running_eval_loss += loss.item()
        eval_accuracy = compute_accuracy(outputs, rotation)
        running_eval_accuracy += eval_accuracy
        if i % 10 == 9:
            print('Eval loss: ' + str(running_eval_loss / 10))
            print('Eval ACC: ' + str(running_eval_accuracy / 10))
            running_eval_loss = 0.0
            running_eval_accuracy = 0.0

print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))

