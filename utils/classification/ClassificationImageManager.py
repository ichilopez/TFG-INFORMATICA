
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
       self.test_dataset = ClassificationDataSet(path=os.path.join(self.main_path,"test"))
       self.val_dataset = ClassificationDataSet(path=os.path.join(self.main_path,"validation"),transform=val_transform)


    def train_dataloader(self):
       return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
       return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def val_dataloader(self):
       return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def configure_optimizers(self):
     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5
     )

     return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss"
        }
    }



    def show_images_with_labels(self, dataloader, title="Train Images", num_images=5):
     imgs, labels = next(iter(dataloader))

     plt.figure(figsize=(12, 4))
     for i in range(min(num_images, len(imgs))):
        plt.subplot(1, num_images, i + 1)

        img = imgs[i].permute(1, 2, 0).squeeze() if imgs[i].ndim == 3 else imgs[i].squeeze()
        plt.imshow(img, cmap='gray')

        plt.title(f"Etiqueta: {labels[i]}", fontsize=10)
        plt.axis('off')

     plt.suptitle(title, fontsize=14)
     plt.tight_layout()
     plt.show()
