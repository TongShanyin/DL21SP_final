import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from torchvision import datasets, transforms, models

from dataloader import CustomDataset, ExtraDataDataset
from contrastive import Encoder, train_transforms, validation_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str) # this is where the final checkpoint will go
parser.add_argument('--model_checkpoint', type=str) # checkpoint for entire model
args = parser.parse_args()

#trainset = CustomDataset(root='/dataset', split='train', transform=train_transforms)
trainset = ExtraDataDataset(root='/dataset', transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

evalset = CustomDataset(root='/dataset', split="val", transform=validation_transforms)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

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

NUM_FEATURE = 2048
#NUM_FEATURE = 512

encoder = Encoder()
classifier = LinearClassifier(NUM_FEATURE)
net = nn.Sequential(encoder, classifier)
net.load_state_dict(torch.load(args.model_checkpoint))
net = net.cuda()

print('SimCLR encoder + linear classifier + finetuning')


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.0002, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140], gamma=0.5)

print('Start Training')
print('use checkpoint:'+args.model_checkpoint)

tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(300):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader):
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

    scheduler.step()
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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "extra_finetune_classifier_epoch" + str(epoch + 1) + ".pth"))
    print("Saved intermediate checkpoint to finetune_classifier_epoch{epoch+1}.pth")


print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))
#print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'classifier_simclr.pth')}")
