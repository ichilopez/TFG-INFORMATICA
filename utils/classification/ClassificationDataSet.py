from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class ClassificationDataSet(Dataset):


    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        all_folders = sorted([
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
        ])
        self.image_folders = all_folders[:923]

    def __len__(self):
        return len(self.image_folders)

    def _read_txt(self, file_path, default="UNKNOWN"):
        if not os.path.exists(file_path):
            return default
        with open(file_path, "r") as f:
            value = f.readline().strip().upper()
        return value if value != "" else default


    def __getitem__(self, idx):
        img_folder = os.path.join(self.path, self.image_folders[idx])

        img_path = os.path.join(img_folder, "1.jpeg")
        label_path = os.path.join(img_folder, "pathology.txt")
        density_path = os.path.join(img_folder, "breast_density.txt")
        view_path = os.path.join(img_folder, "image_view.txt")

        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        # label
        pathology = self._read_txt(label_path)
        if pathology == "BENIGN":
            label = 0
        elif pathology == "MALIGNANT":
            label = 1
        else:
            raise ValueError(f"Etiqueta desconocida en {label_path}: {pathology}")

        density = self._read_txt(density_path)
        view = self._read_txt(view_path)

        meta = torch.tensor([density, view], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return img, meta, label