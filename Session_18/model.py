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


# In[ ]:


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
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        skip = x
        x = self.pool(x)

        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpandingBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], axis=1)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UNet_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet_1, self).__init__()

        self.contract1 = ContractingBlock(3, 64)
        self.contract2 = ContractingBlock(64, 128)
        self.contract3 = ContractingBlock(128, 256)
        self.contract4 = ContractingBlock(256, 512)


                # Center
        self.center = DoubleConv(512, 1024)
        self.expand1 = ExpandingBlock(1024, 512)
        self.expand2 = ExpandingBlock(512, 256)
        self.expand3 = ExpandingBlock(256, 128)
        self.expand4 = ExpandingBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 =  self.contract4(x)
        x = self.center(x)
        #print("center",x.shape)
        x = self.expand1(x, skip4)
        #print("x",x.shape)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)
        x= self.final_conv(x)
        #x=self.sigmoid(x)
        return x 


# ## case 2: MP+Tr+Dice Loss

# In[ ]:


import torch.nn as nn
import torch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        skip = x
        x = self.pool(x)

        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpandingBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], axis=1)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UNet_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet_2, self).__init__()

        self.contract1 = ContractingBlock(3, 64)
        self.contract2 = ContractingBlock(64, 128)
        self.contract3 = ContractingBlock(128, 256)
        self.contract4 = ContractingBlock(256, 512)


                # Center
        self.center = DoubleConv(512, 1024)
        self.expand1 = ExpandingBlock(1024, 512)
        self.expand2 = ExpandingBlock(512, 256)
        self.expand3 = ExpandingBlock(256, 128)
        self.expand4 = ExpandingBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 =  self.contract4(x)
        x = self.center(x)
        #print("center",x.shape)
        x = self.expand1(x, skip4)
        #print("x",x.shape)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)
        x= self.final_conv(x)
        #x=self.sigmoid(x)
        return x 


# ## case 3: Strided + Tran +  BCE

# In[ ]:


import torch.nn as nn
import torch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class ContractingBlock_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingBlock_3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        skip = x

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x, skip

class ExpandingBlock_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpandingBlock_3, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], axis=1)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UNet_3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet_3, self).__init__()

        self.contract1 = ContractingBlock_3(3, 64)
        self.contract2 = ContractingBlock_3(64, 128)
        self.contract3 = ContractingBlock_3(128, 256)
        self.contract4 = ContractingBlock_3(256, 512)


                # Center
        self.center = DoubleConv(512, 1024)
        self.expand1 = ExpandingBlock_3(1024, 512)
        self.expand2 = ExpandingBlock_3(512, 256)
        self.expand3 = ExpandingBlock_3(256, 128)
        self.expand4 = ExpandingBlock_3(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 =  self.contract4(x)
        x = self.center(x)
        #print("center",x.shape)
        x = self.expand1(x, skip4)
        #print("x",x.shape)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)
        x= self.final_conv(x)
        #x=self.sigmoid(x)
        return x 


# ## Case 4: StrConv+Ups+Dice Loss

# In[ ]:


import torch.nn as nn
import torch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class ContractingBlock_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingBlock_4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        skip = x

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x, skip

class ExpandingBlock_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpandingBlock_4, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], axis=1)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UNet_4(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet_4, self).__init__()

        self.contract1 = ContractingBlock_4(3, 64)
        self.contract2 = ContractingBlock_4(64, 128)
        self.contract3 = ContractingBlock_4(128, 256)
        self.contract4 = ContractingBlock_4(256, 512)


                # Center
        self.center = DoubleConv(512, 1024)
        self.expand1 = ExpandingBlock_4(1024, 512)
        self.expand2 = ExpandingBlock_4(512, 256)
        self.expand3 = ExpandingBlock_4(256, 128)
        self.expand4 = ExpandingBlock_4(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 =  self.contract4(x)
        x = self.center(x)
        #print("center",x.shape)
        x = self.expand1(x, skip4)
        #print("x",x.shape)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)
        x= self.final_conv(x)
        #x=self.sigmoid(x)
        return x 


