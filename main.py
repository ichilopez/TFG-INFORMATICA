import yaml
import torch
from utils import dataset 
from models.EfficientNetB0Classifier import EfficientNetB0Classifier
from models.MobileNetV2Classifier import MobileNetV2Classifier
from models.ResNet18Classifier import ResNet18Classifier
from models.YOLOSegmenter import YOLOSegmenter
from models.UNetResNet34Segmenter import UNetResNet34Segmenter
from  models.ViTSmallClassifier import ViTSmallClassifier
import torch
def main(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_dataLoader, test_dataLoader = dataset.getDataLoaders(
        cfg["data"]["study_type"],
        cfg["model"]["model_name"],
        cfg["train"]["batch_size"],
        cfg["data"]["num_workers"]
    )
    
    model = get_model(
    model_name=cfg["model"]["model_name"],
    num_classes=cfg["model"]["num_classes"],   # asegúrate de tenerlo en el YAML
    model_path=cfg["save"]["model_input_path"]
    )
    print("Training...")
    # model.train(model, train_dataLoader, device, cfg["train"]["epochs"], cfg["train"]["learning_rate"],cfg["data"]["study_type"])
    print("Evaluating...")
    # model.evaluate(test_dataLoader)
    print("Saving..")
    # if not cfg["model"]["output_dir"]:
    #print("⚠️ Advertencia: no se especificó 'output_dir', se usará la ruta por defecto.")
    # model.save()  
    #else:
    # model.save(output_dir)
    # print("Modelo guardado exitosamente")



def get_model(model_name: str, num_classes=2, model_path: str = None):

    model_name = model_name.lower()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'mobilenetv2':
        model = MobileNetV2Classifier(num_classes=num_classes, model_path=model_path)

    elif model_name == 'efficientnetb0':
        model = EfficientNetB0Classifier(num_classes=num_classes, model_path=model_path)

    elif model_name == 'resnet18':
        model = ResNet18Classifier(num_classes=num_classes, model_path=model_path)

    elif model_name == 'vitsmall':
        model = ViTSmallClassifier(num_classes=num_classes, model_path=model_path)

    elif model_name == 'yolo':
        model = YOLOSegmenter(model_path=model_path)
    
    elif model_name == 'unetresnet34':
        model = UNetResNet34Segmenter(model_path=model_path)

    else:
        raise ValueError(f"Modelo no reconocido: '{model_name}'. Opciones válidas: "
                         f"'mobilenetv2', 'efficientnetb0', 'resnet18', 'vitsmall', 'yolo','unetresnet34'.")
    return model.to(device)



if __name__ == "__main__":
    main()
