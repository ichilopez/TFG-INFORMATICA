from models import Classifier
from torchvision import models
import torch.nn as nn

class MedViTClassifier(Classifier):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.medvit_small(weights=models.MedViT_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
