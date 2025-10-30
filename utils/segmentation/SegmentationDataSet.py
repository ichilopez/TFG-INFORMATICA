from torch.utils.data import Dataset
from PIL import Image
import torch

class SegmentationDataset(Dataset):
    def __init__(self, data_info, transform=None):
        self.data_info = data_info
        self.transform = transform


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]
        img = Image.open(info['file_path_image']).convert("L")  

        orig_w, orig_h = img.size
        
        if self.transform:
            img = self.transform(img)
        
        _, new_h, new_w = img.shape

        # Extraer coordenadas originales
        x_min = float(info['x_min'])
        y_min = float(info['y_min'])
        x_max = float(info['x_max'])
        y_max = float(info['y_max'])
        x_min = x_min * new_w / orig_w
        x_max = x_max * new_w / orig_w
        y_min = y_min * new_h / orig_h
        y_max = y_max * new_h / orig_h

        coords = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return img, coords
