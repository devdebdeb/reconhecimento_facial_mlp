import os
import cv2
import torch
import numpy as np
from model import MLP

class FacePredictor:
    def __init__(self, model_path, class_names, img_size=48, threshold=0.5):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.threshold   = threshold
        self.img_size    = (img_size, img_size)
        self.model       = MLP(img_size*img_size, len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, image):
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.img_size)
        tensor  = (resized.astype(np.float32) / 127.5) - 1.0
        return torch.FloatTensor(tensor).view(-1).unsqueeze(0).to(self.device)

    def predict(self, image):
        inp = self.preprocess(image)
        with torch.no_grad():
            out   = self.model(inp)
            probs = torch.softmax(out, dim=1)
            conf, idx = probs.max(1)
        return (self.class_names[idx], conf.item()) #if conf > self.threshold else ("Desconhecido", 0.0)


if __name__ == "__main__":
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    model_path  = os.path.join(project_dir, "models", "best_model_fold_1.pth")

    photos_dir  = os.path.join(project_dir, "minhas_fotos")

    predictor = FacePredictor(
        model_path=model_path,
        class_names=["andre", "cesar", "enzo", "will"]
    )

    for name in predictor.class_names:
        filename = f"{name}_teste.jpg"
        img_path = os.path.join(photos_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Erro ao carregar imagem: {img_path}")
            continue

        label, conf = predictor.predict(img)
        print(f"{filename} -> Predição: {label:10s} | Confiança: {conf:.2%}")
