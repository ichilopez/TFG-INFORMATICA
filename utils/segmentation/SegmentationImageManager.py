
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
from utils.segmentation.SegmentationDataset import SegmentationDataset
import torch
import cv2
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





def show_unet_resnet34_samples(train_loader, num_samples=5, recrop_size=(128, 128)):
    shown = 0
    plt.figure(figsize=(10, num_samples * 4))

    for batch in train_loader:
        imgs = batch["image"]
        masks = batch["mask"]

        for i in range(len(imgs)):
            if shown >= num_samples:
                break

            # Convertir tensor a numpy HWC si es necesario
            img = imgs[i]
            mask = masks[i]

            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            if torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy().squeeze()

            # Si CHW -> HWC
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            # Normalizamos para visualizar (opcional)
            img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Buscar bounding box de ROI en la máscara
            ys, xs = np.where(mask > 0.5)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            crop = img_disp[y_min:y_max, x_min:x_max]

            # Escalar recorte para visualización sin alterar datos
            crop_resized = cv2.resize(crop, recrop_size, interpolation=cv2.INTER_NEAREST)

            # ---- Mostrar ----
            plt.subplot(num_samples, 2, 2 * shown + 1)
            plt.imshow(img_disp.squeeze(), cmap='gray')
            plt.title(f"Imagen completa {shown+1}")
            plt.axis("off")

            plt.subplot(num_samples, 2, 2 * shown + 2)
            if crop_resized.ndim == 2 or crop_resized.shape[2] == 1:
                plt.imshow(crop_resized.squeeze(), cmap='gray')
            else:
                plt.imshow(crop_resized)
            plt.title(f"Recorte desde ROI map {shown+1}")
            plt.axis("off")

            shown += 1

        if shown >= num_samples:
            break

    plt.tight_layout()
    plt.show()







        


       




       





