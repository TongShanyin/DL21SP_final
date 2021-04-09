# Autoencoder implementation

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

train_transforms = transforms.Compose([
                                    transforms.Resize((128, 128)),
                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    transforms.RandomRotation(20, resample=Image.BILINEAR),
                                    #transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])

validation_transforms = transforms.Compose([
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(), 
                                    normalize,
                                ])

class ResBlock(torch.nn.Module):

  def __init__(self, in_chan=3, out_chan=3):
    super().__init__()
    self.in_chan = in_chan
    self.out_chan = out_chan
    self.conv1 = torch.nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.conv2 = torch.nn.Conv2d(in_channels=self.out_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.batch_norm1 = torch.nn.BatchNorm2d(num_features=self.out_chan)
    self.batch_norm2 = torch.nn.BatchNorm2d(num_features=self.out_chan)
    
  def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = x + residual
    x = F.relu(x)
    return x

class Encoder(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv3 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv4 = torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv5 = torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), stride=2, padding=(1, 1))

    self.res1 = ResBlock(6, 6)
    self.res2 = ResBlock(12, 12)
    self.res3 = ResBlock(24, 24)
    self.res4 = ResBlock(48, 48)

  def forward(self, x):
    x = self.res1(F.relu(self.conv1(x))) # 6 x 64 x 64
    x = self.res2(F.relu(self.conv2(x))) # 12 x 32 x 32
    x = self.res3(F.relu(self.conv3(x))) # 24 x 16 x 16
    x = self.res4(F.relu(self.conv4(x))) # 48 x 8 x 8
    x = self.conv5(x) # 96 x 4 x 4
    return x

class Decoder(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.deconv1 = torch.nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv2 = torch.nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv3 = torch.nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv4 = torch.nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv5 = torch.nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)

    self.res1 = ResBlock(48, 48)
    self.res2 = ResBlock(24, 24)
    self.res3 = ResBlock(12, 12)
    self.res4 = ResBlock(6, 6)


  def forward(self, x):
    x = self.res1(F.relu(self.deconv1(x))) # 48 x 8 x 8
    x = self.res2(F.relu(self.deconv2(x))) # 24 x 16 x 16
    x = self.res3(F.relu(self.deconv3(x))) # 12 x 32 x 32
    x = self.res4(F.relu(self.deconv4(x))) # 6 x 64 x 64
    x = self.deconv5(x) # 3 x 128 x 128
    return x


