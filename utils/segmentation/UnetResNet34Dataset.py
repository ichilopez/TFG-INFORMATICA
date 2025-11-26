from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class UnetDataset(Dataset):
    def __init__(self, data_info, transform=None):
        self.data_info = data_info
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        img = np.array(Image.open(info['file_path_image']).convert("L"))
        img = np.stack([img] * 3, axis=-1)

        mask = np.array(Image.open(info['roi_mask_path']).convert("L"))
        mask = (mask > 127).astype(np.uint8)

        if all(k in info for k in ['x_min', 'y_min', 'x_max', 'y_max']):
            bboxes = [[
                float(info['x_min']),
                float(info['y_min']),
                float(info['x_max']),
                float(info['y_max'])
            ]]
        else:
            bboxes = []

        if self.transform:
            labels = [0] * len(bboxes)  # o 0 si siempre hay 1 bbox
            transformed = self.transform(image=img, mask=mask, bboxes=bboxes,labels=labels)
            img = transformed['image']
            mask = transformed['mask']
            bboxes = transformed['bboxes']

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return {
            "image": img,
            "mask": mask,
            "bboxes": bboxes,
            "image_path": info['file_path_image'],
            "mask_path": info['roi_mask_path']
        }
