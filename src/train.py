import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import FaceDataset
from model import MLP
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Preparar diretório para salvar modelos
        os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
        
        # Carregar dataset
        self._load_datasets()
        
        # Inicializar modelo
        self.model = MLP(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        # Otimizador e critério
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3
        )
        
    def _load_datasets(self):
        # Dataset base
        base_dataset = FaceDataset(
            self.config['dataset_path'],
            img_size=(48, 48)
        )
        
        # Divisão treino/validação
        train_size = int(self.config['train_ratio'] * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_subset, val_subset = random_split(base_dataset, [train_size, val_size])
        
        # Transformações
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Wrappers para aplicar transformações
        self.train_dataset = self.WrappedDataset(train_subset, train_transform)
        self.val_dataset = self.WrappedDataset(val_subset, val_transform)
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            image_flat, label = self.subset[idx]
            image = Image.fromarray((image_flat.view(48, 48).numpy() * 255).astype(np.uint8))
            return self.transform(image).view(-1), label
            
    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        
        for images, labels in tqdm(self.train_loader, desc="Treinando"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
        return total_loss / len(self.train_loader), total_correct / total_samples
        
    def validate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validando"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
        return total_loss / len(self.val_loader), total_correct / total_samples
        
    def save_model(self, path, epoch, val_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, path)
        
    def train(self):
        best_loss = float('inf')
        early_stop_counter = 0
        
        print(f"Iniciando treinamento na {self.device}")
        print(f"Tamanho do treino: {len(self.train_dataset)}")
        print(f"Tamanho da validação: {len(self.val_dataset)}")
        print(f"Configuração: {self.config}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nÉpoca {epoch + 1}/{self.config['num_epochs']}")
            
            # Treino
            train_loss, train_acc = self.train_epoch()
            
            # Validação
            val_loss, val_acc = self.validate()
            
            # Ajustar LR
            self.scheduler.step(val_loss)
            
            # Salvar melhor modelo
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(self.config['model_save_path'], epoch, val_loss)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            # Early stopping
            if early_stop_counter >= self.config.get('early_stop_patience', 5):
                print(f"Early stopping at epoch {epoch + 1}")
                break
                
            # Log
            print(f"Treino - Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"Validação - Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
            print(f"LR atual: {self.optimizer.param_groups[0]['lr']:.2e}")
            
        print("\nTreinamento concluído!")
        print(f"Melhor loss de validação: {best_loss:.4f}")

if __name__ == "__main__":
    # Configuração
    config = {
        'dataset_path': './processed_dataset',
        'input_size': 48 * 48,
        'hidden_sizes': [1024, 512, 256, 64],
        'num_classes': 4,
        'dropout_rate': 0.3,
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'train_ratio': 0.85,
        'early_stop_patience': 5,
        'model_save_path': './trained_models/best_mlp_model.pth'
    }
    
    # Treinar
    trainer = Trainer(config)
    trainer.train()