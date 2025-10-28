from models import Classifier
from torchvision import models
import torch.nn as nn

class EfficientNetB0Classifier(Classifier):
    def __init__(self, num_classes: int):
        super().__init__()
        # Cargar EfficientNet-B0 preentrenado en ImageNet
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Congelar todas las capas
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Reemplazar la capa final (clasificador)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
