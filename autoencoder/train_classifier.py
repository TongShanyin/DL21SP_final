import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from autoencoder_tightrope import Encoder, Decoder, train_transforms
from classifier import Classifier, LinearClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str) # this is where the final checkpoint will go
parser.add_argument('--encoder_checkpoint', type=str) # checkpoint for pre-trained encoder (first half of the autoencoder)
args = parser.parse_args()

trainset = CustomDataset(root='/dataset', split='train', transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

pretrained_encoder = Encoder()
pretrained_encoder.load_state_dict(torch.load(args.encoder_checkpoint))
classifier = LinearClassifier()
net = nn.Sequential(pretrained_encoder, classifier).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

print('Start Training')
tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(100):
    running_loss = 0.0
    #if steps > 10:
    #    break
    for i, data in enumerate(trainloader):
        steps += 1
        #if steps > 10:
        #    break
        # get the inputs; data is a list of Äinputs, labelsÜ
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(labels[0])
        #print(outputs[0])

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('Ä%d, %5dÜ loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))

os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "net_classifier.pth"))
print(f"Saved checkpoint to äos.path.join(args.checkpoint_dir, 'net_classifier.pth')¨")




