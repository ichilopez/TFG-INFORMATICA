
from torchvision import transforms
from utils.segmentation.SegmentationImageManager import SegmentationImageManager
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from utils.classification.ClassificationImageManager import ClassificationImageManager

def getDataLoaders(study_type, modelName, batch_size, num_workers):
    validate_model_study(study_type, modelName)
    train_transform, test_transform = getTransforms(modelName)
    imageManager = getImageManager(study_type)
    trainDataLoader, testDataLoader = imageManager.getDataLoaders(
        batch_size, num_workers, train_transform, test_transform
    )
    return trainDataLoader, testDataLoader


def getTransforms(modelName: str):

    if modelName.lower() in ["resnet34", "mobilenetv2", "efficientnetb0", "medvit"]:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
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

    elif modelName.lower() == "yolo":
        train_transform = A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=5, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        test_transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    else:
        raise ValueError(
            f"Modelo '{modelName}' no soportado. Usa 'resnet34', 'mobilenetv2', 'efficientnetb0', 'medvit' o 'yolo'.")

    return train_transform, test_transform



   
def getImageManager(study_type):
    if study_type == "segmentationStudy":
        return SegmentationImageManager()
    if study_type == "classificationStudy":
     return ClassificationImageManager()


def validate_model_study(study_type, model_name):
    model_name = model_name.lower()
    if study_type == "classificationStudy" and model_name == "yolo":
        raise ValueError("No puedes usar un modelo YOLO para clasificación.")
    if study_type == "segmentationStudy" and model_name in ["resnet34", "mobilenetv2", "efficientnetb0", "medvit"]:
        raise ValueError("No puedes usar un modelo de clasificación para segmentación.")






