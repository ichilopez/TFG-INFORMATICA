from models.Model import Model
from torchvision import models
import torch.nn as nn


class MobileNetV2Classifier(Model):
    def __init__(self,num_classes=2):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
 
    
    def getModel(self):
        return self.model

