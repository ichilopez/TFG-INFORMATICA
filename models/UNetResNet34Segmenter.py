import torch
import segmentation_models_pytorch as smp
from models.Model import Model

class UNetResNet34Segmenter(Model):
    def __init__(self,device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("Encoder congelado.")

    def getModel(self):
      return self.model

