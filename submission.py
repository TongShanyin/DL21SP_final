# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
from torch import nn
from torch.nn import functional as F

team_id = 6
team_name = "SMC"
email_address = "ccp5804@nyu.edu"


class ResBlock(torch.nn.Module):

  def __init__(self, in_chan=3, out_chan=3, shrink=True):
    super().__init__()
    self.in_chan = in_chan
    self.out_chan = out_chan
    self.shrink = shrink
    self.conv1 = torch.nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=1, padding=(1, 1))
    if shrink:
      self.conv2 = torch.nn.Conv2d(in_channels=self.out_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=2, padding=(1, 1))
    else:
      self.conv2 = torch.nn.Conv2d(in_channels=self.out_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.batch_norm1 = torch.nn.BatchNorm2d(num_features=self.out_chan)
    self.batch_norm2 = torch.nn.BatchNorm2d(num_features=self.out_chan)
    self.batch_norm3 = torch.nn.BatchNorm2d(num_features=self.out_chan)
    self.residual_channel_change = torch.nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(1, 1))
    
  def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    # x = self.conv3(x)
    # x = self.batch_norm3(x)
    #x = F.relu(x)

    if self.shrink:
      residual = F.interpolate(residual, scale_factor=(1/2, 1/2))
    residual = self.residual_channel_change(residual)
    x = x + residual
    
    x = F.relu(x)
    return x

class CNN(torch.nn.Module):

  #LINEAR_SIZE = 1024 * 6 * 6
  LINEAR_SIZE = 512*6*6

  def __init__(self):
    super().__init__()
    self.res1 = ResBlock(3, 9, shrink = False)
    self.res2 = ResBlock(9, 27, shrink = False)
    self.res3 = ResBlock(27, 81)
    self.res4 = ResBlock(81, 243, shrink = False)
    self.res5 = ResBlock(243, 512)
    self.res6 = ResBlock(512, 1024, shrink=False)

    self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
    self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

   # self.linear1 = nn.Linear(self.LINEAR_SIZE, 128)
    #self.linear2 = nn.Linear(128, 800)
    self.linear3 = nn.Linear(self.LINEAR_SIZE, 800)

  def forward(self, x):
    result = x

    result = self.res1(result)
    result = self.pool1(result)
    result = self.res2(result)
    #result = self.pool2(result)
    result = self.res3(result)
    result = self.res4(result)
    result = self.pool3(result)
    result = self.res5(result)
   # result = self.res6(result)

    #print(result.size())
    result = torch.reshape(result, (result.size(0), -1))

    #result = F.relu(self.linear1(result))
    #result = F.sigmoid(self.linear2(result))
    #result = self.linear2(result)
    result = self.linear3(result)
    return result


def get_model():
    return models.resnet18(num_classes=800)
    #return CNN()

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])
