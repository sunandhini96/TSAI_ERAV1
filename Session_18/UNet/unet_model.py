#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
#import ternausnet.models
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

cudnn.benchmark = True


# ## case 1:  MP+Tr+BCE

# In[ ]:


import torch.nn as nn
import torch


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels,pool="none"):
        super(ContractingBlock, self).__init__()

        self.double_conv=DoubleConv(in_channels,out_channels)

        self.pool_layer = get_pooling_layer(pool,out_channels)

    def DoubleConv(self,in_channels,out_channels):
       return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
       
    def get_pooling_layer(self,pool,out_cannels):
        if pool == "max":
          return nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool== "stridedconv":
          return nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1)
        else:
          return nn.Identity()

    def forward(self, x):

        x = self.double_conv(x)

        skip = x
        x = self.pool_layer(x)

        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels,up_sample="transpose"):
        super(ExpandingBlock, self).__init__()

        self.upsample = get_upsampling_layer(in_channels,out_channels,up_sample)
        self.double_conv=DoubleConv(in_channels,out_channels)

    def DoubleConv(self,in_channels,out_channels):
       return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def get_upsampling_layer(self,in_channels,out_channels,up_sample):
      if up_sample=="upsample":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        )
        
      else:
        return nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], axis=1)

        x = self.double_conv(x)


        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,up_sample="transpose",pool="none"):
        super(UNet_1, self).__init__()
        # contracting
        self.contract1 = ContractingBlock(3, 64,pool)
        self.contract2 = ContractingBlock(64, 128,pool)
        self.contract3 = ContractingBlock(128, 256,pool)
        self.contract4 = ContractingBlock(256, 512,pool)


        # Center
        self.center = ContractingBlock(512, 1024) # we are applying only double convolution 

        # expanding
        self.expand1 = ExpandingBlock(1024, 512,up_sample)
        self.expand2 = ExpandingBlock(512, 256,up_sample)
        self.expand3 = ExpandingBlock(256, 128,up_sample)
        self.expand4 = ExpandingBlock(128, 64,up_sample)
 
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 =  self.contract4(x)
        x,_ = self.center(x)
        #print("center",x.shape)
        x = self.expand1(x, skip4)
        #print("x",x.shape)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)
        x= self.final_conv(x)
        #x=self.sigmoid(x)
        return x 


