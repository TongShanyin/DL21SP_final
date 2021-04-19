import os
import torch
import torchvision
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image # PIL is a library to process images



class JigsawDataset(torch.utils.data.Dataset):

    CROP_TRANSFORM = transforms.RandomResizedCrop(96, scale=(0.8, 0.8))

    PERMUTATIONS = [
                    [0, 1, 2, 3],
                    [0, 1, 3, 2],
                    [0, 2, 1, 3],
                    [0, 2, 3, 1],
                    [0, 3, 1, 2],
                    [0, 3, 2, 1],
                    [1, 0, 2, 3],
                    [1, 0, 3, 2],
                    [1, 2, 0, 3],
                    [1, 2, 3, 0],
                    [1, 3, 0, 2],
                    [1, 3, 2, 0],
                    [2, 0, 1, 3],
                    [2, 0, 3, 1],
                    [2, 1, 0, 3],
                    [2, 1, 3, 0],
                    [2, 3, 0, 1],
                    [2, 3, 1, 0],
                    [3, 0, 1, 2],
                    [3, 0, 2, 1],
                    [3, 1, 0, 2],
                    [3, 1, 2, 0],
                    [3, 2, 0, 1],
                    [3, 2, 1, 0],
    ]

    def __init__(self, root, split, pre_jigsaw_transforms, post_jigsaw_transforms):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.pre_jigsaw_transforms = pre_jigsaw_transforms
        self.post_jigsaw_transforms = post_jigsaw_transforms

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_images

    def get_jigsaw_pieces(self, img):
        width, height = img.size
        pieces = []
        for horizontal in [0, 1]:
          for vertical in [0, 1]:
            piece = img.crop((horizontal * (width / 2), vertical * (height / 2), (horizontal + 1) * (width / 2), (vertical + 1) * (height / 2)))
            pieces.append(piece)
        return pieces

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        img = self.pre_jigsaw_transforms(img)
        img = self.CROP_TRANSFORM(img)
        pieces = self.get_jigsaw_pieces(img)
        pieces = [self.post_jigsaw_transforms(p) for p in pieces]

        perm_choice = random.randint(0, 23)
        perm = self.PERMUTATIONS[perm_choice]
        pieces = [pieces[i] for i in perm]

        return pieces, perm_choice

