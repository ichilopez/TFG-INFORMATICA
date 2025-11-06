import timm
import torch
import torch.nn as nn
from models.Classifier import Classifier
import os

class ViTSmallClassifier(Classifier):
    def __init__(self, num_classes: int, model_path: str = None):
        super().__init__()

        self.model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

        if model_path:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

    def save(self, path="weights/medvit.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✅ Pesos guardados correctamente en: {path}")
