from models.Model import Model
from torchvision import models
import torch.nn as nn

class MobileNetV2Classifier(Model):
    def __init__(self, num_classes=2):
        super().__init__()

        # 1️⃣ Cargar modelo preentrenado
        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # 2️⃣ Congelar todos los features
        for param in self.model.features.parameters():
            param.requires_grad = False

        # 3️⃣ Descongelar solo el último bloque
        for param in self.model.features[-1].parameters():  # último bloque
            param.requires_grad = True

        # 4️⃣ Nueva cabeza más simple
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def getModel(self):
        return self.model