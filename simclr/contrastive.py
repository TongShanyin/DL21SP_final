import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
color_jitter = transforms.ColorJitter(0.8, 0.8 , 0.8 , 0.2)

resize_size = 96
#resize_size = 84

train_transforms = transforms.Compose([
                                    transforms.RandomResizedCrop(size=resize_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomApply([color_jitter], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.RandomRotation(25),
                                    #transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])
         
validation_transforms = transforms.Compose([
                                    transforms.Resize(size=resize_size),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module): # resnet50 after average pooling
    def __init__(self):
        super(Encoder, self).__init__()
       # self.encoder = torchvision.models.resnet50()
        self.encoder = torchvision.models.resnet18()
        self.encoder.fc = Identity()

    def forward(self, x):
        x = self.encoder(x)
        return x # resnet50: 2048, resnet 18: 512

class Projector(nn.Module): # MLP with one hidden layer
    def __init__(self,num_feature, num_latent):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(num_feature, num_feature, bias=False)
        self.linear2 = nn.Linear(num_feature, num_latent, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class SimCLR(nn.Module):
    def __init__(self, num_feature, num_latent):
        super(SimCLR, self).__init__()
        self.encoder = Encoder()
        self.proj = Projector(num_feature, num_latent)

    def forward(self, xi, xj):
        zi = self.proj(self.encoder(xi))
        zj = self.proj(self.encoder(xj))
        return zi, zj


class NTXent(nn.Module): # compute the normalized temperature-scaled cross entropy loss
    def __init__(self, batch_size, temperature):
        super(NTXent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, zi, zj):
        z = torch.cat((zi, zj), dim = 0)
        z = F.normalize(z, dim=1) # normalize zi/\|zi\|
        sim = torch.matmul(z, z.T)/self.temperature  # similarity matrix
        mask = torch.eye(2*self.batch_size, dtype=torch.bool).cuda()
        sim = sim[~mask].view(2*self.batch_size,-1) # remove diagonal
        label = torch.cat((torch.arange(self.batch_size-1,2*self.batch_size-1), torch.arange(0,self.batch_size)), dim = 0).cuda()
        loss = nn.CrossEntropyLoss().cuda()
        return loss(sim, label)
