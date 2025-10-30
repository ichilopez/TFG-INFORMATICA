from abc import abstractmethod
from models import Classifier
from torchvision import models
import torch.nn as nn

class ResNet34Classifier(Classifier):
    def __init__(self, num_classes: int):
        super().__init__() 
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V2)
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
