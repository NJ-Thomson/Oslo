#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

train_data = datasets.FashionMNIST(
    root='data',
    transform=transforms.ToTensor(),
    download=True)

test_data = datasets.FashionMNIST(
    root='data',
    transform=transforms.ToTensor(),
    train=False,
    download=True)

print(len(train_data), len(test_data)) 

train_data, val_data = random_split(train_data, [50000, 10000]) 

train_dl = DataLoader(train_data, batch_size=128, shuffle=True)
val_dl = DataLoader(val_data, batch_size=128, shuffle=False)
test_dl = DataLoader(test_data, batch_size=128, shuffle=False)

print(len(train_dl), len(val_dl), len(test_dl))

images, labels = next(iter(train_dl))
print(images.shape, labels.shape)


