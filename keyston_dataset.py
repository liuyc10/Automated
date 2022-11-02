from typing import Optional, Callable, Dict, List, Optional, Tuple

import os
import numpy as np

import torch
from PIL import Image

from torchvision.datasets.vision import VisionDataset


class KCDS(VisionDataset):
    def __init__(self, label_file_path, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(transform=transform, target_transform=target_transform)

        with open(label_file_path, 'r', encoding='UTF-8') as f:
            self.images = list(map(lambda line: line.strip().split(' '), f))

    def __getitem__(self, index):
        path, label = self.images[index]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            image = self.target_transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

