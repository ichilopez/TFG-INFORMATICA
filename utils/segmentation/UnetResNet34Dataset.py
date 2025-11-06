from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import cv2

class UnetResNet34Dataset(Dataset):
    def __init__(self, data_info, transform):
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

        height, width = img.shape[1], img.shape[2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(bboxes) > 0:
            x_min, y_min, x_max, y_max = map(int, bboxes[0])
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, -1)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return {"image": img, "mask": mask}
