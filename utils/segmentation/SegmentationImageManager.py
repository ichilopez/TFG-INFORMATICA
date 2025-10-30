from utils.ImageManager import ImageManager
import pandas as pd
from utils.segmentation.SegmentationDataSet import SegmentationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
path_mass_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_test_set.csv'
path_mass_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_train_set.csv'


class SegmentationImageManager(ImageManager):      
    
    def getDataLoaders(self, batch_size, num_workers, train_transform=None, test_transform=None):
        """Devuelve listas con rutas y coordenadas para train y test."""
        train_data = pd.read_csv(path_mass_train)
        test_data = pd.read_csv(path_mass_test)

        train_info = self.__getPathsAndCoords(train_data)
        test_info = self.__getPathsAndCoords(test_data)

        train_dataset = SegmentationDataset(data_info=train_info, transform=train_transform)
        test_dataset = SegmentationDataset(data_info=test_info, transform=test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self.show_images_with_crops(train_loader)
        return train_loader, test_loader
    

    def __getPathsAndCoords(self, data):
        """Crea una lista de diccionarios con file_path, mask_path y coordenadas."""
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
    
    
    def show_images_with_crops(self, dataloader, num_images=5):

      def denormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(-1,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(-1,1,1)
        return (img_tensor * std + mean).clamp(0,1)

      plt.figure(figsize=(10, 2 * num_images))
      shown = 0

      for imgs, coords in dataloader:
        for i in range(len(imgs)):
            if shown >= num_images:
                break

            # Des-normalizar y pasar a CPU
            img = denormalize(imgs[i].detach().cpu())
            img = img.permute(1,2,0).numpy()  # (H,W,C) para imshow

            h, w, _ = img.shape

            # Ajustar coordenadas
            x_min, y_min, x_max, y_max = coords[i]
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            x_min = max(0, min(x_min, w-1))
            x_max = max(0, min(x_max, w))
            y_min = max(0, min(y_min, h-1))
            y_max = max(0, min(y_max, h))

            if x_max <= x_min or y_max <= y_min:
                continue

            crop = img[y_min:y_max, x_min:x_max]

            # Mostrar original
            plt.subplot(num_images, 2, 2*shown + 1)
            plt.imshow(img)  # 3 canales, no cmap
            plt.title(f"Original {shown+1}")
            plt.axis('off')

            # Mostrar recorte
            plt.subplot(num_images, 2, 2*shown + 2)
            plt.imshow(crop)
            plt.title(f"Recorte {shown+1}")
            plt.axis('off')

            shown += 1

        if shown >= num_images:
            break

      plt.tight_layout()
      plt.show()

