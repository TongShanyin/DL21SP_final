# Please do not change this file.
# We will use this dataloader to benchmark your model.
# If you find a bug, post it on campuswire.

import os
import random
from PIL import Image
import csv

import torch

class CustomDataset(torch.utils.data.Dataset):
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

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)
    
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img), self.labels[idx]

class ExtraDataDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform):
        self.split = 'train'
        self.transform = transform

        self.image_dir = os.path.join(root, self.split)
        label_path = os.path.join(root, f"{self.split}_label_tensor.pt")

        self.num_original_images = len(os.listdir(self.image_dir))
        self.num_images = self.num_original_images + 12800

        self.extra_filenames = []
        with open('request_06.csv', 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.extra_filenames.append(line[0])

        self.extra_labels = torch.load('label_06.pt')

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)
        

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if idx < self.num_original_images:
            with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
                img = Image.open(f).convert('RGB')
                label = self.labels[idx]
        else:
            with open(self.extra_filenames[idx - self.num_original_images], 'rb') as f:
                img = Image.open(f).convert('RGB')
                label = self.extra_labels[idx - self.num_original_images]

        return self.transform(img), label
        
