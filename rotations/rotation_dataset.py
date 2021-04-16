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

class PermutationTransform:
    """Permute the image."""

    def __init__(self, permutation):
      self.permutation = permutation

    def stitch_image(self, top_left, top_right, bot_left, bot_right):
      stitched_image = Image.new('RGB', (96, 96))
      stitched_image.paste(im=top_left, box=(0, 0))
      stitched_image.paste(im=top_right, box=(48, 0))
      stitched_image.paste(im=bot_left, box=(0, 48))
      stitched_image.paste(im=bot_right, box=(48, 48))
      return stitched_image

    def __call__(self, x):
        crops = torchvision.transforms.FiveCrop(size=48)(x)

        # Original image:
        # | 1 | 2 |
        # | 3 | 4 |
        crop1, crop2, crop3, crop4, _ = crops

        if self.permutation == 'rotate_right':
          result = self.stitch_image(top_left=crop3, top_right=crop1, bot_left=crop4, bot_right=crop2)
        elif self.permutation == 'rotate_left':
          result = self.stitch_image(top_left=crop2, top_right=crop4, bot_left=crop1, bot_right=crop3)
        elif self.permutation == 'swap_diag_1':
          result = self.stitch_image(top_left=crop4, top_right=crop2, bot_left=crop3, bot_right=crop1)
        elif self.permutation == 'swap_diag_2':
          result = self.stitch_image(top_left=crop1, top_right=crop3, bot_left=crop2, bot_right=crop4)
        else:
          raise Exception('Unknown permutation ' + str(self.permutation))

        return result


rotation_0 = RotationTransform(angles=[0])
rotation_90 = RotationTransform(angles=[90])
rotation_180 = RotationTransform(angles=[180])
rotation_270 = RotationTransform(angles=[270])

permutation_rotate_right = PermutationTransform(permutation='rotate_right')
permutation_rotate_left = PermutationTransform(permutation='rotate_left')
permutation_swap_diag_1 = PermutationTransform(permutation='swap_diag_1')
permutation_swap_diag_2 = PermutationTransform(permutation='swap_diag_2')

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

        trans_choice = random.randint(0, 3)
        if trans_choice == 0:
          trans = rotation_0
        elif trans_choice == 1:
          trans = rotation_90
        elif trans_choice == 2:
          trans = rotation_180
        elif trans_choice == 3:
          trans = rotation_270
        else:
          raise Exception("Bad trans choice " + str(trans_choice))

        return self.transform(trans(img)), trans_choice
