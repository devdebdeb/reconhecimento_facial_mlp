import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from dataset import FaceDataset
from model import MLP
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.full_dataset = FaceDataset(config['dataset_path'], is_train=True)
        self.model = MLP(48*48, 4).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def train(self):
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.full_dataset.image_paths, self.full_dataset.labels)):
            print(f"\n=== Fold {fold+1}/5 ===")
            
            train_set = Subset(self.full_dataset, train_idx)
            val_set = Subset(self.full_dataset, val_idx)
            
            sampler = self.full_dataset.get_sampler(train_idx)
            
            train_loader = DataLoader(
                train_set,
                batch_size=self.config['batch_size'],
                sampler=sampler,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_set,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            self._reset_model()
            self._train_fold(train_loader, val_loader, fold)

    def _reset_model(self):
        self.model = MLP(48*48, 4).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')

    def _train_fold(self, train_loader, val_loader, fold):
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss, total_correct = 0, 0
            
            # Treino
            for images, labels in tqdm(train_loader, desc=f"Treino Ep.{epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()
            
            # Validação
            val_loss, val_acc = self._validate(val_loader)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), f"./models/best_model_fold_{fold+1}.pth")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= 5:
                    print(f"Early stopping na época {epoch+1}")
                    break
            
            print(f"Época {epoch+1}")
            print(f"  Treino - Loss: {total_loss/len(train_loader):.4f} | Acc: {total_correct/len(train_loader.dataset):.2%}")
            print(f"  Validação - Loss: {val_loss:.4f} | Acc: {val_acc:.2%}\n")

    def _validate(self, val_loader):
        self.model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validando"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()
        return total_loss/len(val_loader), total_correct/len(val_loader.dataset)

if __name__ == "__main__":
    config = {
        'dataset_path': './processed_dataset',
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.0001
    }
    os.makedirs("./models", exist_ok=True)
    trainer = Trainer(config)
    trainer.train()