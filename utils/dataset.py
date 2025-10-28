
from torchvision import transforms
from utils.segmentation.SegmentationDataSet import SegmentationDataSet
from utils.classification.ClassificationDataSet import ClassificationDataSet
from utils.segmentation.SegmentationImageManager import SegmentationImageManager
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.classification.ClassificationImageManager import ClassificationImageManager

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
      transforms.Resize((224, 224)),              
      transforms.Grayscale(num_output_channels=3),
      transforms.ToTensor(),                      
      transforms.Normalize(                      
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
      )
      ])
   
def getImageManager(study_type):
    if study_type == "segmentationStudy":
        return SegmentationImageManager()
    if study_type == "classificationStudy":
     return ClassificationImageManager()

def createDataLoaders(study_type,imageManager,batch_size,num_workers,transform):
    input_train_paths, target_train_paths, input_test_paths,target_test_paths = imageManager.getPaths()

    if study_type == "segmentationStudy":
     train_dataset = SegmentationDataSet(input_paths=input_train_paths,target_paths=target_train_paths,transform=transform)  
     test_dataset = SegmentationDataSet(input_paths=input_test_paths,target_paths=target_test_paths,transform=transform)             
     trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
     testDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if study_type == "classificationStudy":
      train_dataset = ClassificationDataSet(input_paths=input_train_paths,targets=target_train_paths,transform=transform)          
      test_dataset = ClassificationDataSet(input_paths=input_test_paths,targets=target_test_paths,transform=transform) 
      trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
      testDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  
    return trainDataLoader, testDataLoader
 
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

def show_images_with_labels(dataloader, title, num_images=5):
    """
    Muestra algunas imágenes del DataLoader de clasificación con su etiqueta debajo.
    """
    # Obtener un batch del dataloader
    imgs, labels = next(iter(dataloader))

    plt.figure(figsize=(12, 4))
    for i in range(min(num_images, len(imgs))):
        plt.subplot(1, num_images, i + 1)
        
        # Convertir tensor a imagen (permute si es 3 canales)
        img = imgs[i].permute(1, 2, 0).squeeze() if imgs[i].ndim == 3 else imgs[i].squeeze()
        plt.imshow(img, cmap='gray')
        
        # Mostrar etiqueta debajo
        plt.title(f"Etiqueta: {labels[i]}", fontsize=10)
        plt.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()





