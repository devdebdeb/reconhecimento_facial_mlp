import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import FaceDataset
from model import MLP

df = FaceDataset('../processed_dataset')
train_size = int(0.85 * len(df))
train_dataset, val_dataset = random_split(df, [train_size, val_size])