import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CharlieDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.isCharlie = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.isCharlie)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.isCharlie.iloc[idx, 0])
        image = io.imread(img_name)
        isCharlie = self.isCharlie.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, int(isCharlie)