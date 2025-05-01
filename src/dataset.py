import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(48, 48)):
        super(FaceDataset, self).__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.transform = transform
        self.img_size = img_size

        # Validação do diretório
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {self.root_dir}")

        # Processamento mais robusto das classes
        class_dirs = sorted([d for d in os.listdir(self.root_dir) 
                          if os.path.isdir(os.path.join(self.root_dir, d))])
        
        if not class_dirs:
            raise ValueError(f"Nenhuma classe/pasta encontrada em {self.root_dir}")

        # Mapeamento de classes
        for idx, person_name in enumerate(class_dirs):
            self.class_to_idx[person_name] = idx
            self.idx_to_class[idx] = person_name

            person_path = os.path.join(self.root_dir, person_name)
            
            # Processamento de imagens com tratamento de erros
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                try:
                    if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                        
                    # Verificação mais robusta da imagem
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None or img.size == 0:
                        print(f"Imagem inválida: {image_path}")
                        continue
                        
                    self.image_paths.append(image_path)
                    self.labels.append(idx)
                    
                except Exception as e:
                    print(f"Erro ao processar {image_path}: {str(e)}")
                    continue

        if not self.image_paths:
            raise ValueError("Nenhuma imagem válida encontrada no dataset")

        print(f"Dataset carregado com sucesso. Total de imagens: {len(self.image_paths)}")
        print(f"Classes encontradas: {list(self.class_to_idx.keys())}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Carregamento com tratamento de erro
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Falha ao carregar imagem: {image_path}")
                
            image = cv2.resize(image, self.img_size)
            
            # Transformações
            if self.transform:
                image = Image.fromarray(image)
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
                image = transforms.Normalize((0.5,), (0.5,))(image)
            
            return image, label
            
        except Exception as e:
            print(f"Erro ao processar item {idx}: {str(e)}")
            # Retornar um item vazio para evitar quebrar o treinamento
            dummy_img = torch.zeros(1, *self.img_size)
            return dummy_img, -1  # Label inválido

    def get_class_distribution(self):
        """Retorna a distribuição de classes"""
        return {cls: self.labels.count(idx) for cls, idx in self.class_to_idx.items()}