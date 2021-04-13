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
                                    #transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
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

  def __init__(self, channels=3, activation=F.relu):
    super().__init__()
    self.activation = activation
    self.in_chan = channels
    self.out_chan = channels
    self.conv1 = torch.nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.conv2 = torch.nn.Conv2d(in_channels=self.out_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.batch_norm1 = torch.nn.BatchNorm2d(num_features=self.out_chan)
    self.batch_norm2 = torch.nn.BatchNorm2d(num_features=self.out_chan)
    
  def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = x + residual
    x = self.activation(x)
    return x


class Encoder(torch.nn.Module):

  def __init__(self, mode=None):
    super().__init__()
    self.mode = mode

    if self.mode == 'relu':
      self.activation = F.relu
    else:
      self.activation = F.tanh

    if self.mode == 'dropout':
      self.dropout1 = torch.nn.Dropout(p=0.1)
      self.dropout2 = torch.nn.Dropout(p=0.1)
      self.dropout3 = torch.nn.Dropout(p=0.1)
      self.dropout4 = torch.nn.Dropout(p=0.1)

    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(7, 7), stride=2, padding=(3, 3))
    self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv3 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), stride=2, padding=(1, 1))
    self.conv4 = torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), stride=2, padding=(1, 1))

    self.linear1 = nn.Linear(48 * 8 * 8, 48 * 8 * 8)
    self.linear2 = nn.Linear(48 * 8 * 8, 48 * 8 * 8)

    self.res1_1 = ResBlock(6, activation=self.activation)
    self.res1_2 = ResBlock(6, activation=self.activation)
    self.res1_3 = ResBlock(6, activation=self.activation)

    self.res2_1 = ResBlock(12, activation=self.activation)
    self.res2_2 = ResBlock(12, activation=self.activation)
    self.res2_3 = ResBlock(12, activation=self.activation)

    self.res3_1 = ResBlock(24, activation=self.activation)
    self.res3_2 = ResBlock(24, activation=self.activation)
    self.res3_3 = ResBlock(24, activation=self.activation)

    self.res4_1 = ResBlock(48, activation=self.activation)
    self.res4_2 = ResBlock(48, activation=self.activation)
    self.res4_3 = ResBlock(48, activation=self.activation)

  def forward(self, x):
    x = self.activation(self.conv1(x)) # 6 x 64 x 64
    x = self.res1_3(self.res1_2(self.res1_1(x)))
    if self.mode == 'dropout':
      x = self.dropout1(x)

    x = self.activation(self.conv2(x)) # 12 x 32 x 32
    x = self.res2_3(self.res2_2(self.res2_1(x)))
    if self.mode == 'dropout':
      x = self.dropout2(x)

    x = self.activation(self.conv3(x)) # 24 x 16 x 16
    x = self.res3_3(self.res3_2(self.res3_1(x)))
    if self.mode == 'dropout':
      x = self.dropout3(x)

    x = self.activation(self.conv4(x)) # 48 x 8 x 8
    x = self.res4_3(self.res4_2(self.res4_1(x)))
    if self.mode == 'dropout':
      x = self.dropout4(x)

    x = torch.reshape(x, (x.size(0), -1))
    x = self.activation(self.linear1(x))
    x = self.linear2(x)

    return x

class Decoder(torch.nn.Module):

  def __init__(self, mode=None):
    super().__init__()
    self.mode = mode

    if self.mode == 'relu':
      self.activation = F.relu
    else:
      self.activation = F.tanh

    if self.mode == 'dropout':
      self.dropout1 = torch.nn.Dropout(p=0.1)
      self.dropout2 = torch.nn.Dropout(p=0.1)
      self.dropout3 = torch.nn.Dropout(p=0.1)
      self.dropout4 = torch.nn.Dropout(p=0.1)

    self.deconv1 = torch.nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv2 = torch.nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv3 = torch.nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)
    self.deconv4 = torch.nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=(3, 3), stride=2, padding=(1, 1), output_padding=1)

    self.res1_1 = ResBlock(48, activation=self.activation)
    self.res1_2 = ResBlock(48, activation=self.activation)
    self.res1_3 = ResBlock(48, activation=self.activation)

    self.res2_1 = ResBlock(24, activation=self.activation)
    self.res2_2 = ResBlock(24, activation=self.activation)
    self.res2_3 = ResBlock(24, activation=self.activation)

    self.res3_1 = ResBlock(12, activation=self.activation)
    self.res3_2 = ResBlock(12, activation=self.activation)
    self.res3_3 = ResBlock(12, activation=self.activation)

    self.res4_1 = ResBlock(6, activation=self.activation)
    self.res4_2 = ResBlock(6, activation=self.activation)
    self.res4_3 = ResBlock(6, activation=self.activation)

    self.linear1 = nn.Linear(48 * 8 * 8, 48 * 8 * 8)
    self.linear2 = nn.Linear(48 * 8 * 8, 48 * 8 * 8)

  def forward(self, x):
    x = self.activation(self.linear1(x))
    x = self.activation(self.linear2(x))
    x = torch.reshape(x, (-1, 48, 8, 8))
    if self.mode == 'dropout':
      x = self.dropout1(x)

    x = self.res1_3(self.res1_2(self.res1_1(x)))
    x = self.activation(self.deconv1(x)) # 24 x 16 x 16
    if self.mode == 'dropout':
      x = self.dropout2(x)

    x = self.res2_3(self.res2_2(self.res2_1(x)))
    x = self.activation(self.deconv2(x)) # 12 x 32 x 32
    if self.mode == 'dropout':
      x = self.dropout3(x)

    x = self.res3_3(self.res3_2(self.res3_1(x)))
    x = self.activation(self.deconv3(x)) # 6 x 64 x 64
    if self.mode == 'dropout':
      x = self.dropout4(x)

    x = self.res4_3(self.res4_2(self.res4_1(x)))
    x = self.deconv4(x) # 3 x 128 x 128

    return x


