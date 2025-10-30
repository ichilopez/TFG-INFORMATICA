from utils.ImageManager import ImageManager
import os
import pandas as pd
from torch.utils.data import DataLoader
from utils.classification.ClassificationDataSet import ClassificationDataSet
import matplotlib.pyplot as plt
path_mass_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_test_set.csv'
path_mass_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_train_set.csv'

class ClassificationImageManager(ImageManager):
    def getDataLoaders(self, batch_size, num_workers, train_transform=None, test_transform=None):
        train_data = pd.read_csv(path_mass_train)
        test_data = pd.read_csv(path_mass_test)

        train_info = self.__getPathsAndLabels(train_data)
        test_info = self.__getPathsAndLabels(test_data)

        train_dataset = ClassificationDataSet(data_info=train_info, transform=train_transform)
        test_dataset = ClassificationDataSet(data_info=test_info, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.show_images_with_labels(train_loader)

        return train_loader, test_loader

    def __getPathsAndLabels(self, data):
        info_list = []
        for i in range(len(data)):
            try:
                image_path = data.loc[i, 'cropped image file path']
                label = data.loc[i, 'pathology']
                info_list.append({
                    'file_path_image': image_path,
                    'label': label
                })
            except Exception as e:
                print(f"⚠️ Error procesando índice {i}: {e}")

        return info_list
    
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
