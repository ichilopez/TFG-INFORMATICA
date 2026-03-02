from torch.utils.data import Dataset
from PIL import Image
import os

class ClassificationDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        # Obtenemos todas las subcarpetas ordenadas
        self.image_folders = sorted([
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
        ])

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, idx):
        img_folder = os.path.join(self.path, self.image_folders[idx])
        
        # Seleccionamos la imagen según el índice
        if idx < 923:
            img_name = "1.jpeg"
        else:
            img_name = "1_aug.jpeg"
            
        img_path = os.path.join(img_folder, img_name)
        label_path = os.path.join(img_folder, "pathology.txt")

        # Cargar imagen en grayscale
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        # Leer etiqueta
        with open(label_path, "r") as f:
            word = f.readline().strip().upper()
            if word == "BENIGN":
                label = 0
            elif word == "MALIGNANT":
                label = 1
            else:
                raise ValueError(f"Etiqueta desconocida en {label_path}: {word}")

        return img, label