
from torchvision import transforms
from utils.ImageDataSet import ImageDataSet
from utils.SegmentationImageManager import SegmentationImageManager
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
path_meta = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/meta.csv'
path_dicom = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/dicom_info.csv'
path_calcification_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/calc_case_description_test_set.csv'
path_calcification_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/calc_case_description_train_set.csv'
path_mass_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_test_set.csv'
path_mass_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_train_set.csv'

index = 1216

def getDataLoaders(study_type, modelName,batch_size, num_workers):
    transform = getTransform(modelName)
    imageManager = getImageManager(study_type)
    
    return createDataLoaders(study_type,imageManager,batch_size,num_workers,transform)

def getTransform(modelName):
   if modelName == "resnet34":
      return transforms.Compose([
      transforms.Resize((224, 224)),              # Tamaño esperado por ResNet
      transforms.Grayscale(num_output_channels=3),# Convierte 1 canal → 3 canales
      transforms.ToTensor(),                      # Escala [0,255] → [0,1]
      transforms.Normalize(                       # Normalización de ImageNet
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
      )
      ])
   
def getImageManager(study_type):
    if study_type == "segmentationStudy":
        return SegmentationImageManager()

def createDataLoaders(study_type,imageManager,batch_size,num_workers,transform):
    input_train_paths, target_train_paths, input_test_paths,target_test_paths = imageManager.getPaths()
    if study_type == "segmentationStudy":
     train_dataset = ImageDataSet(input_paths=input_train_paths,target_paths=target_train_paths,transform=transform)            # para las imágenes de entradatarget_transform=transform      # para las de salida (puede ser diferente))
     test_dataset = ImageDataSet(input_paths=input_test_paths,target_paths=target_test_paths,transform=transform) 
     trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
     testDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # --- Mostrar algunas imágenes con sus máscaras ---
     def show_images_with_masks(dataloader, title, num_images=5):
      # Obtener un batch del dataloader
      imgs, masks = next(iter(dataloader))

      plt.figure(figsize=(10, 4))
      for i in range(num_images):
        # Imagen
        plt.subplot(2, num_images, i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title(f"Imagen {i+1}")
        plt.axis('off')

        # Máscara / salida
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(masks[i].permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title(f"Máscara {i+1}")
        plt.axis('off')

      plt.suptitle(title, fontsize=14)
      plt.tight_layout()
      plt.show()

    # show_images_with_masks(trainDataLoader, "Ejemplos del conjunto de entrenamiento")
    # show_images_with_masks(testDataLoader, "Ejemplos del conjunto de prueba")

    return trainDataLoader, testDataLoader, 
 





