import os
import yaml
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import lightning as L

from utils.segmentation.SegmentationDataset import SegmentationDataset


class SegmentationImageManager(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, image_size=(256, 256)):
        super().__init__()

        self.save_hyperparameters()

        with open("configs/config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        self.main_path = self.cfg["data"]["main_path"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.prepare_data_per_node = False

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=0,
                shear=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

        self.eval_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

        self.train_dataset = SegmentationDataset(
            images_path=os.path.join(self.main_path, "train"),
            transform=self.train_transform,
            image_size=self.image_size
        )

        self.val_dataset = SegmentationDataset(
            images_path=os.path.join(self.main_path, "validation"),
            transform=self.eval_transform,
            image_size=self.image_size
        )

        self.test_dataset = SegmentationDataset(
            images_path=os.path.join(self.main_path, "test"),
            transform=self.eval_transform,
            image_size=self.image_size
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