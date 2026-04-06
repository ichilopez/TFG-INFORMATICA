import timm
import torch
import torch.nn as nn
from models.Model import Model

class ViTSmallClassifier(Model):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=num_classes
        )

        # Congelar todo
        for param in self.model.parameters():
            param.requires_grad = False

        # Descongelar los últimos bloques transformer
        for param in self.model.blocks[-2:].parameters():
            param.requires_grad = True

        # Descongelar también la normalización final
        for param in self.model.norm.parameters():
            param.requires_grad = True

        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, meta=None):
        # meta se ignora en el modelo unimodal
        return self.model(x)

    def getModel(self):
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)