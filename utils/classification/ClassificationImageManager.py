
from torch.utils.data import DataLoader
from utils.classification.ClassificationDataSet import ClassificationDataSet
import matplotlib.pyplot as plt
from torchvision import transforms
import yaml
import lightning as L
import os
from torchvision import transforms
import torch
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


    
    def prepare_data(self):
       pass

    def setup(self,stage=None):
       image_size = 224
       brightness= 0.05
       contrast = 0.05
       imagenet_mean = [0.485, 0.456, 0.406]
       imagenet_std = [0.229, 0.224, 0.225] # Transform para entrenamiento
       train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # convierte 1 canal → 3 canales
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply(
         [transforms.ColorJitter(brightness=brightness, contrast=contrast)],
         p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

       val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

       self.train_dataset = ClassificationDataSet(path=os.path.join(self.main_path,"train"),transform=train_transform)
       self.test_dataset = ClassificationDataSet(path=os.path.join(self.main_path,"test"),transform=val_transform)
       self.val_dataset = ClassificationDataSet(path=os.path.join(self.main_path,"validation"),transform=val_transform)


    def train_dataloader(self):
       return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
       return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def val_dataloader(self):
       return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

       

    