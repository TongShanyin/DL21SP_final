import os
import torch
import torchvision
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_0 = RotationTransform(angles=[0])
rotation_90 = RotationTransform(angles=[90])
rotation_180 = RotationTransform(angles=[180])
rotation_270 = RotationTransform(angles=[270])

class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        self.num_images = len(os.listdir(self.image_dir))
    
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        rotation_choice = random.randint(0, 3)
        if rotation_choice == 0:
          rot = rotation_0
        elif rotation_choice == 1:
          rot = rotation_90
        elif rotation_choice == 2:
          rot = rotation_180
        elif rotation_choice == 3:
          rot = rotation_270
        else:
          raise Exception("Bad rotation choice " + str(rotation_choice))

        return self.transform(rot(img)), rotation_choice
