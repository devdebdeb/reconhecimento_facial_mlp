# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from dataset import FaceDataset
from model import MLP
from sklearn.metrics import classification_report, confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/best_model_fold_1.pth"
    DATA_DIR   = "./processed_dataset"
    BATCH_SIZE = 32

    # dataset de validação sem augmentação
    val_ds = FaceDataset(DATA_DIR, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # carrega o modelo
    model = MLP(48*48, len(val_ds.class_to_idx)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # coleta as predições
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out   = model(imgs)
            preds = out.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # imprime relatório
    names = list(val_ds.class_to_idx.keys())
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=names))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
