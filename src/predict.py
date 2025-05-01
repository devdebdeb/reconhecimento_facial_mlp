import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import MLP

class FacePredictor:
    def __init__(self, model_path, idx_to_class, input_size=2304, confidence_threshold=0.7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.idx_to_class = idx_to_class
        self.confidence_threshold = confidence_threshold
        self.num_classes = len(idx_to_class)
        
        # Carregar modelo
        self.model = MLP(input_size, [1024, 512, 256, 64], self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # Transformações
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def preprocess_image(self, image):
        """Pré-processa uma imagem para predição"""
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem")
                
        image = cv2.resize(image, (48, 48))
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)
        
    def predict(self, image):
        """Faz a predição para uma imagem"""
        with torch.no_grad():
            inputs = self.preprocess_image(image)
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            max_prob, pred_idx = torch.max(probs, 1)
            
            confidence = max_prob.item()
            if confidence < self.confidence_threshold:
                return "Desconhecido", confidence
                
            return self.idx_to_class[pred_idx.item()], confidence
            
    def predict_batch(self, image_paths):
        """Faz predição para um lote de imagens"""
        results = []
        for path in image_paths:
            try:
                pred, conf = self.predict(path)
                results.append((path, pred, conf))
            except Exception as e:
                results.append((path, f"Erro: {str(e)}", 0.0))
        return results

if __name__ == "__main__":
    # Configurações
    CONFIG = {
        'model_path': './trained_models/best_mlp_model.pth',
        'idx_to_class': {
            0: "Andre",
            1: "Cesar",
            2: "Enzo",
            3: "Will"
        },
        'confidence_threshold': 0.7
    }
    
    # Exemplo de uso
    predictor = FacePredictor(**CONFIG)
    
    # Testar com imagens
    test_images = [
        "minhas_fotos/andre_teste.jpg",
        "minhas_fotos/cesar_teste.jpg", 
        "minhas_fotos/enzo_teste.jpg",
        "minhas_fotos/will_teste.jpg"
    ]
    
    results = predictor.predict_batch(test_images)
    
    for path, pred, conf in results:
        print(f"Imagem: {os.path.basename(path)}")
        print(f"  Predição: {pred} (Confiança: {conf:.2%})")
        print("-" * 40)