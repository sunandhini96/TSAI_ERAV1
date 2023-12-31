from torchvision.datasets.cifar import CIFAR10
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
import albumentations as A
from torchvision.datasets import CIFAR10
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
class CustomResNet(LightningModule):
    def __init__(self, num_classes=10, data_dir=PATH_DATASETS, hidden_size=16, learning_rate=0.05):
        super(CustomResNet, self).__init__()
        self.save_hyperparameters()
        #self.custom_block = CustomBlock(in_channels=64, out_channels=128)
       # Set our init args as class attributes

        # loading the dataset
        self.EPOCHS = 24
        self.num_classes=num_classes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.prep_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.resblock1 = nn.Sequential(
              nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU()
              )


        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )

        self.maxpoollayer = nn.Sequential(nn.MaxPool2d(kernel_size=4,stride = 4))

        self.fclayer = nn.Linear(512, self.num_classes)
    def loss_function(self, pred, target):
        criterion = torch.nn.CrossEntropyLoss()
        
        return criterion(pred, target)


    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer_1(x)
        r1 = self.resblock1(x)
        x = x + r1
        x = self.layer_2(x)
        x = self.layer_3(x)
        r2 = self.resblock2(x)
        x = x + r2
        x = self.maxpoollayer(x)
        x = x.view((x.shape[0],-1))
        x = self.fclayer(x)

        return F.log_softmax(x,dim=-1)
    def get_loss_accuracy(self, batch):
        images, labels = batch
        predictions = self(images)
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == labels).float().mean()
        loss = self.loss_function(predictions, labels)
        
        return loss, accuracy * 100
    
    

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss_accuracy(batch)
        self.log("loss/train", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("acc/train", accuracy, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss_accuracy(batch)
        self.log("loss/val", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("acc/val", accuracy, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        LEARNING_RATE=0.03
        WEIGHT_DECAY=0
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader

        lr_scheduler = OneCycleLR(
          optimizer,
          max_lr=4.79E-02,
          steps_per_epoch=len(dataloader),
          epochs=24,
          pct_start=5/24,
          div_factor=100,
          three_phase=False,
          final_div_factor=100,
          anneal_strategy='linear'
        )

        scheduler = {"scheduler": lr_scheduler, "interval" : "step"}

        return [optimizer], [scheduler]
