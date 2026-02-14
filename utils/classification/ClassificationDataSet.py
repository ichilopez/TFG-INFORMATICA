from torch.utils.data import Dataset
from PIL import Image
import os

class ClassificationDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_files = os.listdir(path)  # lista de carpetas dentro de path

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # ruta completa a la carpeta del caso
        img_folder = os.path.join(self.path, self.image_files[idx])

        # ruta a la imagen y al label
        img_path = os.path.join(img_folder, "1.jpeg")
        label_path = os.path.join(img_folder, "pathology.txt")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        with open(label_path, "r") as f:
          word = f.readline().strip().upper()
          if word == "BENIGN":
           label = 0
          elif word == "MALIGNANT":
           label = 1
          else:
           raise ValueError(f"Etiqueta desconocida en {label_path}: {word}")
        return img, label