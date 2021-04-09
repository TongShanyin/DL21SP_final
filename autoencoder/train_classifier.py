import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from autoencoder import Encoder, Decoder, train_transforms, EncoderSmall, DecoderSmall
from classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str) # this is where the final checkpoint will go
parser.add_argument('--encoder_checkpoint', type=str) # checkpoint for pre-trained encoder (first half of the autoencoder)
args = parser.parse_args()

trainset = CustomDataset(root='/dataset', split='train', transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

evalset = CustomDataset(root='/dataset', split="val", transform=train_transforms)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

pretrained_encoder = EncoderSmall()
pretrained_encoder.load_state_dict(torch.load(args.encoder_checkpoint))
#classifier = Classifier()
classifier = models.resnet18(num_classes=800)
net = nn.Sequential(pretrained_encoder, classifier).cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

print('Start Training')
tic = time.perf_counter()

net.train()
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of inputs, labels
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    #print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in evalloader:
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {(100 * correct / total):.2f}%")

os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "net_classifier.pth"))
print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_classifier.pth')}")




