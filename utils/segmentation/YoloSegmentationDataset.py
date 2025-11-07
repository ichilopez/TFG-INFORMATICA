from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class YoloDataset(Dataset):
    """
    Dataset para modelos YOLO — devuelve imagen y bounding boxes.
    Las transformaciones de Albumentations ajustan automáticamente las bboxes.
    """
    def __init__(self, data_info, transform=None):
        self.data_info = data_info
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        img = np.array(Image.open(info['file_path_image']).convert("L"))
        img = np.stack([img] * 3, axis=-1)

        bbox = [
            float(info['x_min']),
            float(info['y_min']),
            float(info['x_max']),
            float(info['y_max'])
        ]
        bboxes = [bbox]
        labels = [0] 

        if self.transform:
            transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']

        return {
            "image": img,
            "bboxes": bboxes,
            "image_path": info['file_path_image']
        }
