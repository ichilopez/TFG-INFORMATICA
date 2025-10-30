
from torchvision import transforms
from utils.segmentation.SegmentationImageManager import SegmentationImageManager

from utils.classification.ClassificationImageManager import ClassificationImageManager

def getDataLoaders(study_type, modelName,batch_size, num_workers):
    train_transform,test_transform= getTransforms(modelName)
    imageManager = getImageManager(study_type)
    trainDataLoader, testDataLoader = imageManager.getDataLoaders(batch_size,num_workers,train_transform,test_transform)
    return trainDataLoader, testDataLoader 


from torchvision import transforms

def getTransforms(modelName: str, for_debug: bool = False):
 
    if modelName == "resnet34":
        if for_debug:
            # Transforms fijos para visualización
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = train_transform
        else:
            # Transforms para entrenamiento
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    elif modelName in ["mobilenetv2", "efficientnetb0", "medvit"]:
        if for_debug:
            # Fijo para visualización
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = train_transform
        else:
            # Transform para entrenamiento con augmentations
            train_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    else:
        raise ValueError(f"Modelo '{modelName}' no soportado. Usa 'resnet34', 'mobilenetv2' o 'efficientnetb0'.")

    return train_transform, test_transform


   
def getImageManager(study_type):
    if study_type == "segmentationStudy":
        return SegmentationImageManager()
    if study_type == "classificationStudy":
     return ClassificationImageManager()








