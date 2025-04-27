import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import FaceDataset
from model import MLP

df = FaceDataset('../processed_dataset')
train_size = int(0.85 * len(df))
val_size = len(df) - train_size
train_dataset, val_dataset = random_split(df, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch=32, shuffle=False)

input_size = 2304
hidden_sizes = [512, 128]
num_classes = 4
model = MLP(input_size, hidden_sizes, num_classes, dropout_rate=0.3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
