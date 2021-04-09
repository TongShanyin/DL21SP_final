# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
from torch import nn
from torch.nn import functional as F
from autoencoder_tightrope import Encoder, validation_transforms
from classifier import LinearClassifier

team_id = 6
team_name = "SMC"
email_address = "ccp5804@nyu.edu"


def get_model():
    encoder = Encoder()
    classifier = LinearClassifier()
    return nn.Sequential(encoder, classifier).cuda()
    #return CNN()

eval_transform = validation_transforms
