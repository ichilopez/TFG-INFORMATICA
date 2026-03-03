from models.Model import Model
from torchvision import models
import torch.nn as nn

class MobileNetV2Classifier(Model):
    def __init__(self, num_classes=2, unfreeze_last_blocks=3):
        super().__init__()
        
        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # 🔹 1️⃣ Congelar todo
        for param in self.model.features.parameters():
            param.requires_grad = False

        # 🔹 2️⃣ Descongelar últimos bloques correctamente
        feature_blocks = list(self.model.features.children())
        for block in feature_blocks[-unfreeze_last_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # 🔹 3️⃣ Nueva cabeza
        in_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def getModel(self):
        return self.model