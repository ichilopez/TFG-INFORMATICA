from utils.ImageManager import ImageManager
import pandas as pd
from utils.segmentation.YoloSegmentationDataset import YoloDataset
from torch.utils.data import DataLoader
from utils.segmentation.UnetResNet34Dataset import UnetDataset
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import yaml


class SegmentationImageManager(ImageManager):      
    
    def getDataLoaders(self, batch_size, num_workers, model_name):
        
        with open("configs/config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        self.path_mass_train = self.cfg["data"]["train_csv"]
        self.path_mass_val = self.cfg["data"]["val_csv"]
        self.path_mass_test = self.cfg["data"]["test_csv"]

        train_data = pd.read_csv(self.path_mass_train)
        val_data = pd.read_csv(self.path_mass_val)
        test_data = pd.read_csv(self.path_mass_test)

        if model_name == "yolo":
            train_info = self.__getPathsAndBBoxes(train_data)
            test_info = self.__getPathsAndBBoxes(test_data)
            val_info = self.__getPathsAndBBoxes(val_data)

            train_transform = A.Compose([
             A.Resize(640, 640),
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=5, p=0.5),
             A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
             ToTensorV2()
            ])

            test_transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])

            train_dataset = YoloDataset(data_info=train_info, transform=train_transform)
            val_dataset = YoloDataset(data_info= val_info, transform= test_transform)
            test_dataset = YoloDataset(data_info=test_info, transform=test_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            show_yolo_samples(train_loader)


        elif model_name == "unetresnet34":
          train_info = self.__getImageAndROIMaskPaths(train_data)
          test_info  = self.__getImageAndROIMaskPaths(test_data)
          val_info = self.__getImageAndROIMaskPaths(val_data)

          train_transform = A.Compose([
           A.Resize(256, 256),
           A.HorizontalFlip(p=0.5),
           A.RandomRotate90(p=0.5),
           A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=10, p=0.5),
           A.Normalize(mean=(0.485, 0.485, 0.485),
                    std=(0.229, 0.229, 0.229)),
           ToTensorV2()
          ],
          bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

          test_transform = A.Compose([
           A.Resize(256, 256),
           A.Normalize(mean=(0.485, 0.485, 0.485),
                    std=(0.229, 0.229, 0.229)),
            ToTensorV2()
           ],
          bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
  
          train_dataset = UnetDataset(data_info=train_info, transform=train_transform)
          test_dataset = UnetDataset(data_info=test_info, transform=test_transform)
          val_dataset = UnetDataset(data_info=val_info, transform=test_transform)
            
          train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
          test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
          val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
          show_unet_resnet34_samples(train_loader)
        else:
               raise ValueError(f"Modelo desconocido: {model_name}")
        
        return train_loader,val_loader,test_loader


    def __getPathsAndBBoxes(self, data):
        info_list = []
        for i in range(len(data)):
            try:
                image_path = data.loc[i, 'image file path']
                x_min = data.loc[i, 'x_min']
                y_min = data.loc[i, 'y_min']
                x_max = data.loc[i, 'x_max']
                y_max = data.loc[i, 'y_max']

                info_list.append({
                    'file_path_image': image_path,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                })
            except Exception as e:
                print(f"⚠️ Error procesando índice {i}: {e}")
        return info_list


    def __getImageAndROIMaskPaths(self, data):
        info_list = []
        for i in range(len(data)):
            try:
                image_path = data.loc[i, 'image file path']
                roi_mask_path = data.loc[i, 'ROI mask file path']

                info_list.append({
                    'file_path_image': image_path,
                    'roi_mask_path': roi_mask_path
                })
            except Exception as e:
                print(f"⚠️ Error procesando índice {i}: {e}")
        return info_list
    
    import matplotlib.pyplot as plt


def show_unet_resnet34_samples(train_loader, num_samples=1):
    shown = 0
    plt.figure(figsize=(10, num_samples * 3))

    for batch in train_loader:
        imgs = batch["image"]
        masks = batch["mask"]

        for i in range(len(imgs)):
            if shown >= num_samples:
                break

            # Convertir tensores a numpy
            img = imgs[i].detach().cpu().numpy()
            mask = masks[i].detach().cpu().numpy().squeeze()

            # Reordenar canales (C,H,W) → (H,W,C)
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))

            # Normalizar
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Calcular bounding box del ROI
            ys, xs = np.where(mask > 0.5)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Generar recorte
            crop = img[y_min:y_max, x_min:x_max]

            # Mostrar original
            plt.subplot(num_samples, 2, 2 * shown + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"Imagen completa {shown+1}")
            plt.axis("off")

            # Mostrar recorte
            plt.subplot(num_samples, 2, 2 * shown + 2)
            plt.imshow(crop.squeeze(), cmap='gray')
            plt.title(f"Recorte desde ROI map {shown+1}")
            plt.axis("off")

            shown += 1

        if shown >= num_samples:
            break

    plt.tight_layout()
    plt.show()


def show_yolo_samples(dataloader, num_images=1):
    plt.figure(figsize=(10, 3 * num_images))
    shown = 0

    for batch in dataloader:
        imgs = batch["image"]
        masks = batch["mask"]
        batch_size = len(imgs)

        for i in range(batch_size):
            if shown >= num_images:
                break

            # Convertir tensores a numpy
            img = imgs[i].detach().cpu().numpy()
            mask = masks[i].detach().cpu().numpy().squeeze()

            # Reordenar canales (C, H, W) → (H, W, C)
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))

            # Normalizar a [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Crear superposición con la máscara
            overlay = img.copy()
            if overlay.shape[-1] == 1:
                overlay = np.repeat(overlay, 3, axis=-1)
            overlay[mask > 0.5] = [1.0, 0.0, 0.0]  # rojo = ROI

            # Mostrar original
            plt.subplot(num_images, 2, 2 * shown + 1)
            plt.imshow(img.squeeze(), cmap="gray")
            plt.title(f"Imagen {shown + 1}")
            plt.axis("off")

            # Mostrar con máscara
            plt.subplot(num_images, 2, 2 * shown + 2)
            plt.imshow(overlay)
            plt.title(f"Máscara superpuesta {shown + 1}")
            plt.axis("off")

            shown += 1

        if shown >= num_images:
            break

    plt.tight_layout()
    plt.show()

