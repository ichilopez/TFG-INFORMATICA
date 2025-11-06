from utils.ImageManager import ImageManager
import pandas as pd
from utils.segmentation.YoloSegmentationDataset import YoloSegmentationDataset
from torch.utils.data import DataLoader
from utils.segmentation.UnetResNet34Dataset import UnetResNet34Dataset
import matplotlib.pyplot as plt
import numpy as np

path_mass_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_test_set.csv'
path_mass_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_train_set.csv'


class SegmentationImageManager(ImageManager):      
    
    def getDataLoaders(self, batch_size, num_workers,model_name, train_transform=None, test_transform=None):
        if train_transform is None or test_transform is None:
            raise ValueError("SegmentationDataset requiere Albumentations")
        
        train_data = pd.read_csv(path_mass_train)
        test_data = pd.read_csv(path_mass_test)

        train_info = self.__getPathsAndCoords(train_data)
        test_info = self.__getPathsAndCoords(test_data)

        if model_name == "yolo":
         train_dataset = YoloSegmentationDataset(data_info=train_info, transform=train_transform)
         test_dataset = YoloSegmentationDataset(data_info=test_info, transform=test_transform)
         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
         show_yolo_images_with_crops(train_loader)
        elif  model_name == "unetresnet34":
         train_dataset = UnetResNet34Dataset(data_info=train_info, transform=train_transform)
         test_dataset = UnetResNet34Dataset(data_info=test_info, transform=test_transform) 
         show_unet_convnext_images_with_crops(train_loader)
         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
       
        return train_loader, test_loader

    def __getPathsAndCoords(self, data):
        info_list = []
        
        for i in range(len(data)):
            try:
                image_path = data.loc[i, 'image file path']
                x_min = data.loc[i, 'x_min'] if 'x_min' in data.columns else None
                y_min = data.loc[i, 'y_min'] if 'y_min' in data.columns else None
                x_max = data.loc[i, 'x_max'] if 'x_max' in data.columns else None
                y_max = data.loc[i, 'y_max'] if 'y_max' in data.columns else None

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
    
    
def show_yolo_images_with_crops(dataloader, num_images=5):
    plt.figure(figsize=(10, 2 * num_images))
    shown = 0

    for imgs, coords in dataloader:
        for i in range(len(imgs)):
            if shown >= num_images:
                break

            img = imgs[i].detach().cpu()

            # Si tiene 3 canales idénticos, colapsar a 2D para mostrar
            if img.shape[0] == 3:
                img_show = img[0]  # tomar el primer canal
            else:
                img_show = img[0]

            img_show = img_show.numpy()

            h, w = img_show.shape

            # Ajustar coordenadas
            x_min, y_min, x_max, y_max = coords[i]
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            x_min = max(0, min(x_min, w-1))
            x_max = max(0, min(x_max, w))
            y_min = max(0, min(y_min, h-1))
            y_max = max(0, min(y_max, h))

            if x_max <= x_min or y_max <= y_min:
                continue

            crop = img_show[y_min:y_max, x_min:x_max]

            # Mostrar original
            plt.subplot(num_images, 2, 2*shown + 1)
            plt.imshow(img_show, cmap='gray')
            plt.title(f"Original {shown+1}")
            plt.axis('off')

            # Mostrar recorte
            plt.subplot(num_images, 2, 2*shown + 2)
            plt.imshow(crop, cmap='gray')
            plt.title(f"Recorte {shown+1}")
            plt.axis('off')

            shown += 1

        if shown >= num_images:
            break

    plt.tight_layout()
    plt.show()

def show_unet_convnext_images_with_crops(dataloader, num_images=5):
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

            # Reorganizar canales si es necesario
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)
            elif img.ndim == 2:
                img = np.expand_dims(img, -1)

            # Normalizar la imagen para visualización
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Crear superposición de máscara en rojo
            overlay = img.copy()
            if overlay.shape[-1] == 1:
                overlay = np.repeat(overlay, 3, axis=-1)
            overlay[mask > 0.5] = [1.0, 0.0, 0.0]  # píxeles en rojo

            # Mostrar lado a lado
            plt.subplot(num_images, 2, 2 * shown + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"Imagen {shown + 1}")
            plt.axis("off")

            plt.subplot(num_images, 2, 2 * shown + 2)
            plt.imshow(overlay)
            plt.title(f"Máscara superpuesta {shown + 1}")
            plt.axis("off")

            shown += 1

        if shown >= num_images:
            break

    plt.tight_layout()
    plt.show()

