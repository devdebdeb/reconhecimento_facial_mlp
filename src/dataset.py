import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        super(FaceDataset, self).__init__()  

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.root_dir = os.path.join(project_root, root_dir)

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        for idx, person_name in enumerate(sorted(os.listdir(self.root_dir))):
            person_path = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_path):
                continue

            self.class_to_idx[person_name] = idx
            self.idx_to_class[idx] = person_name

            for image_name in os.listdir(person_path):
                self.image_paths.append(os.path.join(person_path, image_name))
                self.labels.append(idx)

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,))  
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = self.transform(image)  
        image = image.view(-1)  

        return image, label

if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split

    dataset = FaceDataset("../processed_dataset")
    print("Total de imagens:", len(dataset))
    print("Rótulos encontrados:", dataset.class_to_idx)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    for images, labels in train_loader:
        print("Shape do batch:", images.shape)  
        print("Primeiro rótulo do batch:", labels[0])
        break
