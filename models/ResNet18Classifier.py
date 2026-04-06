import torch
import torch.nn as nn
from torchvision import models
from models.Model import Model

class ResNet18Classifier(Model):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Congelar todo
        for param in self.model.parameters():
            param.requires_grad = False

        # Descongelar últimos bloques reales de ResNet
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Si quieres afinar un poco más, puedes descomentar también esto:
        # for param in self.model.layer3.parameters():
        #     param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
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