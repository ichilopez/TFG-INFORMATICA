import torch
import torch.nn as nn
from torchvision import models
from models.Classifier import Classifier
import os

class ResNet18Classifier(Classifier):
    def __init__(self, num_classes=2, model_path=None):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        if model_path:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
    
    def save(self, path="weights/resnet18.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            torch.save(self.model.state_dict(), path)
            print(f"Pesos guardados correctamente en: {path}")
        except Exception as e:
            print(f"Error guardando el modelo: {e}")
