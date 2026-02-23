from models.Model import Model
from torchvision import models
import torch.nn as nn

class EfficientNetB0Classifier(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
        for param in self.model.parameters():
            param.requires_grad = False
            
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)


    def getModel(self):
        return self.model


