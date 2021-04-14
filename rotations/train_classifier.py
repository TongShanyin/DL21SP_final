import os
import argparse

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from dataloader import CustomDataset

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str) # this is where the final checkpoint will go
parser.add_argument('--alexnet_checkpoint', type=str) # checkpoint for pre-trained alexnet
args = parser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])

validation_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])

trainset = CustomDataset(root='/dataset', split='train', transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=192, shuffle=True, num_workers=2)

evalset = CustomDataset(root='/dataset', split="val", transform=validation_transforms)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=192, shuffle=False, num_workers=2)

class LinearClassifier(torch.nn.Module):

  LINEAR_SIZE = 6400

  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(self.LINEAR_SIZE, 2048)
    self.linear2 = nn.Linear(2048, 1024)
    self.linear3 = nn.Linear(1024, 800)

    self.dropout1 = torch.nn.Dropout(p=0.5)
    self.dropout2 = torch.nn.Dropout(p=0.5)
    self.bn1 = torch.nn.BatchNorm1d(self.LINEAR_SIZE)
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

print('CONV4, SMALL CLASSIFIER, DROPOUT+MoreBN')
alexnet = torchvision.models.alexnet(pretrained=False)
alexnet.classifier[6] = torch.nn.Linear(4096, 4)
alexnet.load_state_dict(torch.load(args.alexnet_checkpoint))
for param in alexnet.parameters():
    param.requires_grad = False
#for layer in alexnet.features[:9]:
#    for param in layer.parameters():
#        param.requires_grad = False
feature_extractor = alexnet.features[:9]
classifier = LinearClassifier()
net = nn.Sequential(feature_extractor, classifier)

net = net.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

print('Start Training')
print('use checkpoint'+args.alexnet_checkpoint)

tic = time.perf_counter()

net.train()
steps = 0
for epoch in range(50):
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

    print(f"[{epoch+1}] Validation loss: {validation_loss/130:.3f}, Accuracy: {(100 * correct / total):.2f}%")
    if epoch % 10 == 9: # save every 10 epochs
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "rotation_classifier"+args.alexnet_checkpoint[-2:]+f"_epoch{epoch+1}.pth"))
        #print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_classifier.pth')}")



print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))
print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'rotation_classifier.pth')}")

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




