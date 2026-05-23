from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class ClassificationDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        # Lista las carpetas de estudios disponibles
        all_folders = sorted([
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
        ])

        # Limita el número de muestras usadas
        self.image_folders = all_folders[:916]

    def __len__(self):
        return len(self.image_folders)

    def _read_file(self, file_path):
        # Comprueba que el fichero existe y no está vacío
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No existe el fichero: {file_path}")

        with open(file_path, "r") as f:
            value = f.readline().strip()

        if value == "":
            raise ValueError(f"Fichero vacío: {file_path}")

        return value

    def __getitem__(self, idx):
        img_folder = os.path.join(self.path, self.image_folders[idx])

        img_path = os.path.join(img_folder, "1.jpeg")
        label_path = os.path.join(img_folder, "pathology.txt")
        density_path = os.path.join(img_folder, "breast_density.txt")
        view_path = os.path.join(img_folder, "image_view.txt")

        # Carga la imagen en escala de grises
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        # Convierte la etiqueta textual a clase numérica
        pathology = self._read_file(label_path).upper()
        if pathology == "BENIGN":
            label = 0
        elif pathology == "MALIGNANT":
            label = 1
        else:
            raise ValueError(f"Etiqueta desconocida en {label_path}: {pathology}")

        # Lee los metadatos asociados a la imagen
        density_str = self._read_file(density_path)
        view_str = self._read_file(view_path)

        try:
            density = int(density_str)
        except Exception:
            raise ValueError(f"breast_density inválido en {density_path}: '{density_str}'")

        try:
            view = int(view_str)
        except Exception:
            raise ValueError(f"image_view inválido en {view_path}: '{view_str}'")

        # Valida que los metadatos estén dentro del rango esperado
        if density not in [0, 1, 2, 3]:
            raise ValueError(f"breast_density fuera de rango en {density_path}: {density}")

        if view not in [0, 1]:
            raise ValueError(f"image_view fuera de rango en {view_path}: {view}")

        # Devuelve imagen, metadatos y etiqueta en formato tensor
        meta = torch.tensor([density, view], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return img, meta, label