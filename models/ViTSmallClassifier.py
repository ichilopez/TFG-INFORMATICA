import timm
import torch.nn as nn
from models.Model import Model

class ViTSmallClassifier(Model):
    def __init__(self,num_classes=2):
        super().__init__()
        self.model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)

        for param in self.model.parameters():
            param.requires_grad = False
            
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def getModel(self):
        return self.model

