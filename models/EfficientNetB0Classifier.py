from models.Model import Model
from torchvision import models
import torch
import torch.nn as nn
import os

class EfficientNetB0Classifier(Model):
    def __init__(self, num_classes=2, model_path: str = None):
        super().__init__()
        if model_path:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        else:
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)


    def getModel(self):
        return self.model

<<<<<<< HEAD
    def getModel(self):
        return self.model

       
=======
>>>>>>> recuperar_cambios
    def save(self, path="weights/efficientnetb0.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.model.save(path)
            print(f"✅ Modelo YOLO guardado correctamente en: {path}")
        except AttributeError:
            torch.save(self.model.state_dict(), path)
            print(f"✅ Pesos guardados manualmente en: {path}")

