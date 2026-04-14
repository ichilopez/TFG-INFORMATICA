import os
import yaml
import cv2
import numpy as np
from PIL import Image
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.classification.ClassificationDataSet import ClassificationDataSet


class ApplyCLAHE(object):
    """
    Aplica CLAHE para resaltar texturas y bordes dentro del recorte de la lesión.
    Muy útil para distinguir masas en tejidos densos.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe)


class ClassificationImageManager(L.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()

        with open("configs/config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        self.main_path = self.cfg["data"]["main_path"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data_per_node = False

    def setup(self, stage=None):
        image_size = 256

        # Medias y desviaciones estándar de ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ApplyCLAHE(clip_limit=2.0),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ApplyCLAHE(clip_limit=2.0),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.train_dataset = ClassificationDataSet(
            path=os.path.join(self.main_path, "train"),
            transform=self.train_transform
        )

        self.val_dataset = ClassificationDataSet(
            path=os.path.join(self.main_path, "validation"),
            transform=self.val_transform
        )

        self.test_dataset = ClassificationDataSet(
            path=os.path.join(self.main_path, "test"),
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )