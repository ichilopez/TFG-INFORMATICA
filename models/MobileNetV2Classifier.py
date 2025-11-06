from models.Classifier import Classifier
from torchvision import models
import torch
import torch.nn as nn
import os

class MobileNetV2Classifier(Classifier):
    def __init__(self, num_classes = 2, model_path: str = None):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        if model_path:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)

    def save(self, path="weights/mobilenetv2.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.model.save(path)
            print(f"✅ Modelo YOLO guardado correctamente en: {path}")
        except AttributeError:
            torch.save(self.model.model.state_dict(), path)
            print(f"✅ Pesos guardados manualmente en: {path}")
