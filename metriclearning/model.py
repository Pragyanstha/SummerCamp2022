import os
import time
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import ThresholdReducer


class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc = nn.Sequential(
            nn.Linear(25600, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x