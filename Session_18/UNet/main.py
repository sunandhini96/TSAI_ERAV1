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
from utils import DiceLoss
cudnn.benchmark = True


# In[ ]:


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


# In[ ]:


def train(train_loader, model, criterion, optimizer, epoch, params,train_losses):
    metric_monitor = MetricMonitor()
    model.train()
    train_loss=0
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        train_loss+=loss.item()
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
    train_losses.append(train_loss)
    return train_losses

# In[ ]:


def validate(val_loader, model, criterion, epoch, params,test_losses):
    metric_monitor = MetricMonitor()
    model.eval()
    test_loss=0
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            test_loss+=loss.item()
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
        test_losses.append(test_loss)
        #print("Test loss :", test_loss)
        return test_losses


# In[ ]:


def create_model(params):
    model = UNet()
    model = model.to(params["device"])
    return model


# In[ ]:


def train_and_validate(model, train_dataset, val_dataset, params,loss_fn):
    train_losses=[]
    test_losses=[]
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    criterion = loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    for epoch in range(1, params["epochs"] + 1):
        train_losses=train(train_loader, model, criterion, optimizer, epoch, params,train_losses)
        test_losses=validate(val_loader, model, criterion, epoch, params,test_losses)
    return model,train_losses,test_losses


# In[ ]:


def predict(model, params, test_dataset, batch_size):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=params["num_workers"], pin_memory=True,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_masks = (probabilities >= 0.5).float() * 1
            predicted_masks = predicted_masks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                predicted_masks, original_heights.numpy(), original_widths.numpy()
            ):
                predictions.append((predicted_mask, original_height, original_width))
    return predictions


# In[ ]:


# def train_and_validate_1(model, train_dataset, val_dataset, params):
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=params["batch_size"],
#         shuffle=True,
#         num_workers=params["num_workers"],
#         pin_memory=True,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=params["batch_size"],
#         shuffle=False,
#         num_workers=params["num_workers"],
#         pin_memory=True,
#     )
#     criterion = DiceLoss( ).to(params["device"])
#     optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
#     for epoch in range(1, params["epochs"] + 1):
#         train(train_loader, model, criterion, optimizer, epoch, params)
#         validate(val_loader, model, criterion, epoch, params)
#     return model


# In[ ]:
import matplotlib.pyplot as plt

def plot_curves(train_losses, test_losses, xlabel="Epoch", ylabel="Loss", title="Training and Testing Loss Curves"):
    """
    Plot training and testing loss curves.

    Args:
        train_losses (list): List of training losses for each epoch.
        test_losses (list): List of testing losses for each epoch.
        xlabel (str): Label for the x-axis (default: "Epoch").
        ylabel (str): Label for the y-axis (default: "Loss").
        title (str): Title for the plot (default: "Training and Testing Loss Curves").
    """
    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, test_losses, label="Testing Loss", marker='*')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# In[ ]:





# In[ ]:




