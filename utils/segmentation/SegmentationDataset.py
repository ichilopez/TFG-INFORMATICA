import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, images_path, transform=None, image_size=(256, 256)):
        self.images_path = images_path
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(images_path)
            if os.path.isdir(os.path.join(images_path, f))
        ])

        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]

        img_path = os.path.join(self.images_path, img_file, "0.jpeg")
        mask_path = os.path.join(self.images_path, img_file, "2.jpeg")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarizar máscara por si viene con grises
        mask = (mask > 127).astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float()

        return image, mask


