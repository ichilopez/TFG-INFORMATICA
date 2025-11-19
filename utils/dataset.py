

from utils.segmentation.SegmentationImageManager import SegmentationImageManager
from utils.classification.ClassificationImageManager import ClassificationImageManager

def getDataLoaders(study_type, modelName, batch_size, num_workers):
    validate_model_study(study_type, modelName)
    imageManager = getImageManager(study_type)
    trainDataLoader, valDataLoader, testDataLoader = imageManager.getDataLoaders(
        batch_size, num_workers,modelName)
    return trainDataLoader, valDataLoader, testDataLoader


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






