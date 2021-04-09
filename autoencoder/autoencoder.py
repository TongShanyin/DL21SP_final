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
                                    transforms.Resize((256, 256)),
                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    transforms.RandomRotation(20, resample=Image.BILINEAR),
                                    #transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])

validation_transforms = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(), 
                                    normalize,
                                ])

class Encoder(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=50, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv3 = torch.nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv4 = torch.nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv5 = torch.nn.Conv2d(in_channels=50, out_channels=3, kernel_size=(3, 3), stride=1, padding=(1, 1))
  
  def forward(self, x):
    x = self.conv1(x) # 3 x 128 x 128
    x = F.relu(x)
    x = self.conv2(x) # 10 x 64 x 64
    x = F.relu(x)
    x = self.conv3(x) # 50 x 32 x 32
    x = F.relu(x)
    x = self.conv4(x) # 50 x 16 x 16
    x = F.relu(x)
    x = self.conv5(x) # 3 x 16 x 16
    return x

class Decoder(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.deconv1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv2 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv3 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv4 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)

  def forward(self, x):
    x = self.deconv1(x) # 10 x 32 x 32
    x = F.relu(x)
    x = self.deconv2(x) # 10 x 64 x 64
    x = F.relu(x)
    x = self.deconv3(x) # 10 x 128 x 128
    x = F.relu(x)
    x = self.deconv4(x) # 3 x 256 x 256
    return x


class EncoderSmall(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv4 = torch.nn.Conv2d(in_channels=10, out_channels=3, kernel_size=(3, 3), stride=2, padding=(1, 1))

  def forward(self, x):
    x = self.conv1(x) # 3 x 128 x 128
    x = F.relu(x)
    x = self.conv2(x) # 3 x 64 x 64
    x = F.relu(x)
    x = self.conv3(x) # 10 x 32 x 32
    x = F.relu(x)
    x = self.conv4(x) # 10 x 16 x 16
    return x

class DecoderSmall(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.deconv1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv2 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv3 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv4 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)

  def forward(self, x):
    x = self.deconv1(x) # 3 x 32 x 32
    x = F.relu(x)
    x = self.deconv2(x) # 3 x 64 x 64
    x = F.relu(x)
    x = self.deconv3(x) # 3 x 128 x 128
    x = F.relu(x)
    x = self.deconv4(x) # 3 x 256 x 256
    return x

