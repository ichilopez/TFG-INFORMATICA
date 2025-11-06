
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
        batch_size, num_workers,modelName, train_transform, test_transform
    )
    return trainDataLoader, testDataLoader


def getTransforms(modelName: str):

    if modelName.lower() in ["resnet18", "mobilenetv2", "efficientnetb0", "vitsmall"]:
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

    elif modelName == "unetresnet34":
     train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2()
     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

     test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2()
     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        raise ValueError(
            f"❌ Modelo '{modelName}' no soportado. Usa uno de: "
            f"'resnet18', 'mobilenetv2', 'efficientnetb0', 'vitsmall', 'yolo', 'unetresnet34'.")

    return train_transform, test_transform

def getImageManager(study_type):
    if study_type == "segmentationStudy":
        return SegmentationImageManager()
    if study_type == "classificationStudy":
     return ClassificationImageManager()


def validate_model_study(study_type: str, model_name: str):
    study_type = study_type.lower().strip()
    model_name = model_name.lower().strip()

    valid_studies = ["classificationstudy", "segmentationstudy"]
    classification_models = ["resnet18", "mobilenetv2", "efficientnetb0", "vitsmall"]
    segmentation_models = ["yolo", "unetresnet34"]

    if study_type not in valid_studies:
        raise ValueError(f"❌ Tipo de estudio inválido: '{study_type}'. "
                         f"Debe ser uno de: {valid_studies}")

    valid_models = classification_models + segmentation_models
    if model_name not in valid_models:
        raise ValueError(f"❌ Modelo inválido: '{model_name}'. "
                         f"Debe ser uno de: {valid_models}")
    if study_type == "classificationstudy" and model_name not in classification_models:
        raise ValueError(f"❌ El modelo '{model_name}' no puede usarse para clasificación.")
    if study_type == "segmentationstudy" and model_name not in segmentation_models:
        raise ValueError(f"❌ El modelo '{model_name}' no puede usarse para segmentación.")

    print(f"✅ Asociación válida: {study_type} ↔ {model_name}")






