# Feel free to modifiy this file.

from torchvision import models, transforms

team_id = 6
team_name = "SMC"
email_address = "ccp5804@nyu.edu"

def get_model():
    return models.resnet18(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])
