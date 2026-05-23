import torch
import segmentation_models_pytorch as smp
from models.Model import Model


class UNetResNet34Segmenter(Model):
    def __init__(self, device="cuda"):
        super().__init__()

        # Usa GPU si está disponible
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # U-Net con encoder ResNet34 preentrenado
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

        # Congela el encoder completo
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Ojo: aquí layer4 también queda congelada
        for param in self.model.encoder.layer4.parameters():
            param.requires_grad = False

        print("Encoder congelado.")

    def getModel(self):
        return self.model