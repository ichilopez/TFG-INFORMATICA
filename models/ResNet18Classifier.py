
import torch.nn as nn
from torchvision import models
from models.Model import Model

class ResNet18Classifier(Model):
    def __init__(self,num_classes=2):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def getModel(self):
        return self.model
       
