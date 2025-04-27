import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import FaceDataset
from model import MLP

df = FaceDataset('processed_dataset')
train_size = int(0.85 * len(df))
val_size = len(df) - train_size
train_dataset, val_dataset = random_split(df, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

input_size = 2304
hidden_sizes = [512, 128]
num_classes = 4
model = MLP(input_size, hidden_sizes, num_classes, dropout_rate=0.3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(torch.float32)
        labels = labels.long()

        optimizer.zero_grad()             # Zera gradiente 
        outputs = model(inputs)           # Forward
        loss = criterion(outputs, labels) # Loss
        loss.backward()                   # Backpropagation
        optimizer.step()                  # Atualiza pesos

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(torch.float32)
            labels = labels.long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1} / {num_epochs}], Loss Treino: {avg_train_loss:.4f}, Loss Val: {avg_val_loss:.4f}, Acurácia Val: {val_accuracy:.2}%")

save_path = '../trained_models/mlp_face_recognition.pth'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)

print(f"Modelo salvo em: {save_path}")
