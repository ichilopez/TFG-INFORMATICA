from models.Model import Model
from torchvision import models
import torch.nn as nn
import torch

class MobileNetV2Classifier(Model):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        if in_channels != 3:
            old_conv = self.model.features[0][0]  # Conv2d original (3 canales)
            self.model.features[0][0] = nn.Conv2d(
                in_channels, 
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
        
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