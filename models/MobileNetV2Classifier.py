from models.Model import Model
from torchvision import models
import torch.nn as nn

class MobileNetV2Classifier(Model):
    def __init__(self, num_classes=2):
        super().__init__()
        # Cargar el base model
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features

        # 1. Congelar y descongelar (tu lógica actual)
        for param in self.features.parameters():
            param.requires_grad = False
        for i in range(16, len(self.features)):
            for param in self.features[i].parameters():
                param.requires_grad = True

        # 2. Bloque de Atención SE (Squeeze-and-Excitation)
        in_channels = 1280  # MobileNetV2 termina en 1280 canales
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

        # 3. Cabeza Clasificadora mejorada
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Aplicamos atención: multiplicamos los canales por su importancia calculada
        atencion = self.attention(x)
        x = x * atencion 
        x = self.classifier(x)
        return x

    def getModel(self):
        # Ahora devolvemos 'self' porque este objeto es el nn.Module completo
        return self