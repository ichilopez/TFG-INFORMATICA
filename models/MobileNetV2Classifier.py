from models.Model import Model
from torchvision import models
import torch.nn as nn


class MobileNetV2Classifier(Model):
    def __init__(self,num_classes=2):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes) 
        
    
    def getModel(self):
        return self.model

