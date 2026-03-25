from models.Model import Model
from torchvision import models
import torch.nn as nn

class EfficientNetB0Classifier(Model):
    def __init__(self, num_classes=2):
        super().__init__()
        # 1. Cargar pesos por defecto (ImageNet)
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 2. Congelar inicialmente todo
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 3. Descongelar los últimos 2 bloques de 'features'
        # Esto permite que el modelo aprenda texturas médicas (bordes, densidades)
        # sin perder el conocimiento general de las capas iniciales.
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True
            
        # 4. Clasificador robusto con BatchNorm y SiLU
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512), # Estabiliza el aprendizaje en datos médicos
            nn.SiLU(),           # Mantiene la consistencia con EfficientNet
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def getModel(self):
        return self.model


