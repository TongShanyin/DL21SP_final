# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
from torch import nn
from torch.nn import functional as F
# from autoencoder_tightrope import Encoder, validation_transforms
from classifier import LinearClassifier

team_id = 6
team_name = "SMC"
email_address = "ccp5804@nyu.edu"

# copy Encoder code from autoencoder_tightrope
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


class Classifier(torch.nn.Module):

  LINEAR_SIZE = 27 * 16 * 16

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.conv2 = torch.nn.Conv2d(in_channels=9, out_channels=27, kernel_size=(3, 3), stride=1, padding=(1, 1))
    self.linear1 = nn.Linear(self.LINEAR_SIZE, 800)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = torch.reshape(x, (x.size(0), -1))
    x = F.relu(self.linear1(x))
    return x


# For use with tightrope autoencoder
class LinearClassifier(torch.nn.Module):

  LINEAR_SIZE = 96 * 4 * 4

  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(self.LINEAR_SIZE, self.LINEAR_SIZE)
    self.linear2 = nn.Linear(self.LINEAR_SIZE, self.LINEAR_SIZE)
    self.linear3 = nn.Linear(self.LINEAR_SIZE, self.LINEAR_SIZE)
    self.linear4 = nn.Linear(self.LINEAR_SIZE, 800)

  def forward(self, x):
    x = torch.reshape(x, (x.size(0), -1))
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = self.linear4(x)
    return x

# copy validation_transforms code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

#train_transforms = transforms.Compose([
#                                    transforms.Resize((128, 128)),
#                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
#                                    transforms.RandomRotation(20, resample=Image.BILINEAR),
#                                    #transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
#                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
#                                    normalize,
#                                ])

validation_transforms = transforms.Compose([
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    normalize,
                                ])


def get_model():
    encoder = Encoder()
    classifier = LinearClassifier()
    return nn.Sequential(encoder, classifier).cuda()
    #return CNN()

eval_transform = validation_transforms
