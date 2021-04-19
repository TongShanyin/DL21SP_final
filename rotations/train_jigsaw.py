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

from jigsaw_dataset import JigsawDataset


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

pre_jigsaw_transforms = transforms.Compose([
                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    transforms.RandomRotation(25),
])

post_jigsaw_transforms = transforms.Compose([
                                    transforms.Resize(96),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
])


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

trainset = JigsawDataset(root='/dataset', split="unlabeled", pre_jigsaw_transforms=pre_jigsaw_transforms, post_jigsaw_transforms=post_jigsaw_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=192, shuffle=True, num_workers=2)

evalset = JigsawDataset(root='/dataset', split="val", pre_jigsaw_transforms=pre_jigsaw_transforms, post_jigsaw_transforms=post_jigsaw_transforms)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=192, shuffle=False, num_workers=2)

feature_extractor = torchvision.models.alexnet(pretrained=False)
feature_extractor.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
)
feature_extractor = feature_extractor.cuda()

projector = nn.Linear(4096, 1024)
projector = projector.cuda()

predictor = nn.Sequential(
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 24),
  )
predictor = predictor.cuda()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([
                             {'params': feature_extractor.parameters()},
                             {'params': projector.parameters()},
                             {'params': predictor.parameters()}
                             ],
                            lr=0.01, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

os.makedirs(args.checkpoint_dir, exist_ok=True)

def compute_accuracy(outputs, perm):
  predictions = torch.argmax(outputs, dim=1)
  correct = (predictions == perm)
  return sum(correct).item() / list(correct.size())[0]


print('Start Training')
tic = time.perf_counter()


for epoch in range(100):
    feature_extractor.train()
    projector.train()
    predictor.train()

    if (epoch % 5 == 0) and (epoch != 0):
        torch.save(feature_extractor.state_dict(), os.path.join(args.checkpoint_dir, "jigsaw_ep_%s" % str(epoch)))
        print("Saved intermediate checkpoint to jigsaw_ep_%s" % str(epoch))
        tac = time.perf_counter()
        print("Time elapsed: " + str(tac - tic))

    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(trainloader):
        inputs, perm = data
        inputs = [i.cuda() for i in inputs]
        perm = perm.cuda()

        outputs = [projector(feature_extractor(i)) for i in inputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = predictor(outputs)

        loss = criterion(outputs, perm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        accuracy = compute_accuracy(outputs, perm)
        running_accuracy += accuracy
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10), flush=True)
            print('[%d, %5d] ACC: %.3f' % (epoch + 1, i + 1, running_accuracy / 10), flush=True)
            running_loss = 0.0
            running_accuracy =0.0
    
    feature_extractor.eval()
    projector.eval()
    predictor.eval()
    running_eval_loss = 0.0
    running_eval_accuracy = 0.0
    for i, data in enumerate(evalloader):
        inputs, perm = data
        inputs = [i.cuda() for i in inputs]
        perm = perm.cuda()

        outputs = [projector(feature_extractor(i)) for i in inputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = predictor(outputs)

        loss = criterion(outputs, perm)
        running_eval_loss += loss.item()
        eval_accuracy = compute_accuracy(outputs, perm)
        running_eval_accuracy += eval_accuracy
        if i % 10 == 9:
            print('Eval loss: ' + str(running_eval_loss / 10))
            print('Eval ACC: ' + str(running_eval_accuracy / 10))
            running_eval_loss = 0.0
            running_eval_accuracy = 0.0

print('Finished Training')
toc = time.perf_counter()
print('Time elapsed: ' + str(toc - tic))

