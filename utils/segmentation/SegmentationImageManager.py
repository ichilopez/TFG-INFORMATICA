import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.ImageManager import ImageManager
from utils.segmentation.YoloSegmentationDataset import YoloDataset
from utils.segmentation.UnetResNet34Dataset import UnetDataset
import torch
import cv2


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
            val_info = self.__getPathsAndBBoxes(val_data)
            test_info = self.__getPathsAndBBoxes(test_data)

            train_transform = A.Compose(
                [
                    A.Resize(640, 640),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['labels'])
            )

            test_transform = A.Compose(
                [
                    A.Resize(640, 640),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['labels'])
            )

            # Dataset y DataLoader


            train_dataset = YoloDataset(data_info=train_info)
            val_dataset = YoloDataset(data_info=val_info, transform=test_transform)
            test_dataset = YoloDataset(data_info=test_info, transform=test_transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            show_yolo_crops(train_loader, num_samples=5)

           

        elif model_name == "unetresnet34":
            # Preparar info para U-Net
            train_info = self.__getImageAndROIMaskPaths(train_data)
            val_info = self.__getImageAndROIMaskPaths(val_data)
            test_info = self.__getImageAndROIMaskPaths(test_data)

            # Transformaciones U-Net
            train_transform = A.Compose(
                [
                    A.Resize(256, 256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                    A.Normalize(mean=(0.485, 0.485, 0.485), std=(0.229, 0.229, 0.229)),
                    ToTensorV2()
                ],bbox_params=A.BboxParams(
                 format='pascal_voc',
                 label_fields=['labels'],
                 check_each_transform=False  # evita errores de validación de longitud
               ))
            test_transform = A.Compose(
                [
                    A.Resize(256, 256),
                    A.Normalize(mean=(0.485, 0.485, 0.485), std=(0.229, 0.229, 0.229)),
                    ToTensorV2()
                ]
            )

            train_dataset = UnetDataset(data_info=train_info, transform=train_transform)
            val_dataset = UnetDataset(data_info=val_info, transform=test_transform)
            test_dataset = UnetDataset(data_info=test_info, transform=test_transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            show_unet_resnet34_samples(train_loader,num_samples=5)

        else:
            raise ValueError(f"Modelo desconocido: {model_name}")

        return train_loader, val_loader, test_loader

    def __getPathsAndBBoxes(self, data):
        info_list = []
        for i in range(len(data)):
            try:
                image_path = data.loc[i, 'image file path']
                x_center = float(data.loc[i, 'x_center'])
                y_center = float(data.loc[i, 'y_center'])
                width = float(data.loc[i, 'width'])
                height = float(data.loc[i, 'height'])

                info_list.append({
                    'file_path_image': image_path,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
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




def yolo_to_voc(yolo_bbox, img_w, img_h):
    """Convierte [xc, yc, w, h] normalizados a px [xmin, ymin, xmax, ymax]"""
    xc, yc, w, h = yolo_bbox

    bw = w * img_w
    bh = h * img_h

    x_min = int((xc * img_w) - bw / 2)
    y_min = int((yc * img_h) - bh / 2)
    x_max = int(x_min + bw)
    y_max = int(y_min + bh)

    return x_min, y_min, x_max, y_max


def show_yolo_crops(dataloader, num_samples=5):
    """Muestra imagen completa + recorte usando bbox YOLO normalizada"""
    shown = 0
    plt.figure(figsize=(10, num_samples * 4))

    for batch in dataloader:
        images = batch["image"]
        bboxes = batch["bboxes"]

        for img, bbox_list in zip(images, bboxes):
            if shown >= num_samples:
                break

            # --- Convertir imagen a HWC sin alterar valores ---
            if torch.is_tensor(img):
                img = img.detach().cpu()
                if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW → HWC
                    img_np = img.permute(1, 2, 0).numpy()
                else:  # Ya HWC
                    img_np = img.numpy()
            else:
                img_np = img  # NumPy HWC

            h, w = img_np.shape[:2]

            # Suponemos una sola bbox por imagen
            bbox = bbox_list[0]  # tensor([xc, yc, w, h])
            if torch.is_tensor(bbox):
                bbox = bbox.tolist()  # convertir a lista de floats

            # Convertir YOLO → VOC
            x_min, y_min, x_max, y_max = yolo_to_voc(bbox, w, h)

            # Recorte seguro
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            crop = img_np[y_min:y_max, x_min:x_max]

            # ---- Mostrar ----
            plt.subplot(num_samples, 2, 2 * shown + 1)
            plt.imshow(img_np)
            plt.title(f"Imagen completa {shown+1}")
            plt.axis("off")

            plt.subplot(num_samples, 2, 2 * shown + 2)
            plt.imshow(crop)
            plt.title("Recorte desde bbox")
            plt.axis("off")

            shown += 1

        if shown >= num_samples:
            break

    plt.tight_layout()
    plt.show()



        


       




       





