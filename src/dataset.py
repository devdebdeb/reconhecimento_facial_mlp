import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, WeightedRandomSampler

class FaceDataset(Dataset):
    def __init__(self, root_dir, img_size=(48, 48), is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {"andre": 0, "cesar": 1, "enzo": 2, "will": 3}
        self._load_data()
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if is_train else lambda x: x,
            transforms.RandomRotation(10) if is_train else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _load_data(self):
        print(f"\nCarregando dataset de: {self.root_dir}")
        for class_name in self.class_to_idx:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Aviso: Pasta '{class_name}' não encontrada.")
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Total de imagens carregadas: {len(self.image_paths)}")
        if len(self.image_paths) == 0:
            raise ValueError("Nenhuma imagem válida encontrada!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img.view(-1), label

    def get_sampler(self, indices):
        labels = [self.labels[i] for i in indices]
        class_counts = np.bincount(labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)