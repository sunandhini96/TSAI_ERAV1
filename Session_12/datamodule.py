import os
import pytorch_lightning as pl
import torch
import albumentations as A
from torchvision.datasets import CIFAR10
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

# import pytorch_lightning as pl
# from pytorch_lightning.datasets import CIFAR10
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader



class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        self.classes = dataset.classes

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = np.array(image)
        transformed = self.transforms(image=image)
        image = transformed['image']

        return image, target

    def __len__(self):
        return len(self.dataset)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, train_transforms, val_transforms, test_transforms):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean=(0.4914, 0.4822, 0.4465)
        self.std=(0.2470, 0.2435, 0.2616)
        # self.train_transforms = train_transforms
        # self.val_transforms = val_transforms
        # self.test_transforms = test_transforms
    
    def setup(self, stage=None):
        self.train_transforms = A.Compose([
        A.Normalize(mean=self.mean, std=self.std, always_apply=True),
        A.PadIfNeeded(
        min_height=40,
        min_width=40
    ),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=self.mean, mask_fill_value=None),
        ToTensorV2()
    ])
        self.test_transforms = A.Compose([
        A.Normalize(mean=self.mean, std=self.std),
        ToTensorV2()
    ])

        # Assign train/val datasets 
        if stage == "fit" or stage is None:
            self.train_data = AlbumentationsDataset(CIFAR10(self.data_dir, train=True),  self.train_transforms)
            self.val_data = AlbumentationsDataset(CIFAR10(self.data_dir, train=False), self.test_transforms)

        # Assign test dataset
        if stage == "test" or stage is None:
            self.test_data = AlbumentationsDataset(CIFAR10(self.data_dir, train=False), self.test_transforms)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()
    
