from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class YoloDataset(Dataset):
    def __init__(self, data_info, transform=None):
        self.data_info = data_info
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        with Image.open(info['file_path_image']) as im:
            im = im.convert("RGB")
            img = np.array(im)

        # Coordenadas YOLO normalizadas
        x_center = float(info['x_center'])
        y_center = float(info['y_center'])
        width = float(info['width'])
        height = float(info['height'])

        # Albumentations necesita lista de bboxes
        bboxes = [[x_center, y_center, width, height]]
        labels = [0]

        # Transformación
        if self.transform:
            transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']   # sigue siendo lista de listas

        # Convertimos a tensor (4,) → pero LO GUARDAMOS como lista
        bbox_tensor = torch.tensor(bboxes[0], dtype=torch.float32)

        return {
            "image": img,
            "bboxes": [bbox_tensor],   # SIEMPRE lista de cajas
            "image_path": info['file_path_image']
        }
