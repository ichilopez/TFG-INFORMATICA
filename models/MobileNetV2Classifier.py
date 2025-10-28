
from models import Classifier
from torchvision import models
import torch.nn as nn

class MobileNetV2Classifier(Classifier):
    def __init__(self, num_classes: int):
        super().__init__()
        # Cargar modelo preentrenado en ImageNet
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Congelar todas las capas
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Reemplazar la capa final (clasificador)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
