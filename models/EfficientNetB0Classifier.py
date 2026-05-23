import torch
import torch.nn as nn
from torchvision import models
from models.Model import Model


class EfficientNetB0Classifier(Model):
    def __init__(self, num_classes=2):
        super().__init__()

        # Carga EfficientNet-B0 con pesos preentrenados en ImageNet
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Congela inicialmente todos los parámetros del modelo
        for param in self.model.parameters():
            param.requires_grad = False

        # Se reentrenan solo las últimas capas convolucionales
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True

        # Sustituye el clasificador original por uno adaptado al problema
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, meta=None):
        # La metadata no se usa en el modelo unimodal
        return self.model(x)

    def getModel(self):
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)