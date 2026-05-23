from models.Model import Model
from torchvision import models
import torch
import torch.nn as nn


class MobileNetV2Classifier(Model):
    def __init__(self, num_classes=2):
        super().__init__()

        # Backbone preentrenado
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features

        # Congela el extractor de características
        for param in self.features.parameters():
            param.requires_grad = False

        # Ajuste fino de los últimos bloques
        for i in range(16, len(self.features)):
            for param in self.features[i].parameters():
                param.requires_grad = True

        # Bloque de atención SE
        in_channels = 1280
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Clasificador final
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

    def forward(self, x, meta=None):
        # Modelo unimodal: no usa metadatos
        x = self.features(x)

        # Repondera los canales más relevantes
        attention = self.attention(x)
        x = x * attention

        x = self.classifier(x)
        return x

    def getModel(self):
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)