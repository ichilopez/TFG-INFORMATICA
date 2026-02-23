
import yaml
from torch.utils.data import DataLoader
from utils.segmentation.SegmentationDataset import SegmentationDataset
import torch
import lightning as L
import os 


class SegmentationImageManager(L.LightningDataModule):

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
        train_dataset = SegmentationDataset(images_path=os.path.join(self.main_path,"train"))
        val_dataset = SegmentationDataset(images_path=os.path.join(self.main_path,"validation"))
        test_dataset = SegmentationDataset(images_path=os.path.join(self.main_path,"train"))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        return self.train_loader
    
    def test_dataloader(self):
        return self.test_loader
    
    def val_dataloader(self):
        return self.val_loader
    
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










        


       




       





