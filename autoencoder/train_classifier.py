import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from autoencoder_tightrope import Encoder, Decoder, train_transforms, validation_transforms
from classifier import Classifier, LinearClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str) # this is where the final checkpoint will go
parser.add_argument('--encoder_checkpoint', type=str) # checkpoint for pre-trained encoder (first half of the autoencoder)
args = parser.parse_args()

trainset = CustomDataset(root='/dataset', split='train', transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

evalset = CustomDataset(root='/dataset', split="val", transform=validation_transforms)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)


pretrained_encoder = Encoder()
pretrained_encoder.load_state_dict(torch.load(args.encoder_checkpoint))

classifier = LinearClassifier()
net = nn.Sequential(pretrained_encoder, classifier).cuda()

criterion = nn.CrossEntropyLoss()

print('Encoder: FROZEN WEIGHTS')
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

print('Start Training')
print('use checkpoint'+args.encoder_checkpoint)

tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(30):
    net.train()
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
        if i % 10 == 9:    #print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    net.eval()
    correct = 0
    total = 0
    validation_loss = 0.0
    with torch.no_grad():
        for data in evalloader:
            images, labels = data

            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"[{epoch+1}] Validation loss: {validation_loss/100:.3f}, Accuracy: {(100 * correct / total):.2f}%")
    if epoch % 10 == 9: # save every 10 epochs
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "frozen_encoder_classifier"+args.encoder_checkpoint[-2:]+f"_epoch{epoch+1}.pth"))
        #print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_classifier.pth')}")



print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))
print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'frozen_encoder_classifier.pth')}")

#net.eval()
#correct = 0
#total = 0
#with torch.no_grad():
#    for data in evalloader:
#        images, labels = data

#        images = images.cuda()
#        labels = labels.cuda()

#        outputs = net(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()

#print(f"Accuracy: {(100 * correct / total):.2f}%")

#os.makedirs(args.checkpoint_dir, exist_ok=True)
#torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "net_classifier.pth"))
#print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_classifier.pth')}")




