from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, data_info, transform):
        """
        Dataset para detección de lesiones en mamografías (YOLO).
        Args:
            data_info: lista de diccionarios con:
                - file_path_image
                - x_min, y_min, x_max, y_max
            transform: albumentations.Compose con bbox_params='pascal_voc'
        """
        self.data_info = data_info
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]
        img = np.array(Image.open(info['file_path_image']).convert("L"))
        img = np.stack([img]*3, axis=-1)
        bbox = [float(info['x_min']), float(info['y_min']),
                float(info['x_max']), float(info['y_max'])]
        bboxes = [bbox]
        labels = [0]  

        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        bboxes = transformed['bboxes']

        coords = torch.tensor(bboxes[0], dtype=torch.float32)

        return img, coords
