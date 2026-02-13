import torch
import segmentation_models_pytorch as smp
import os
from models.Model import Model

class UNetResNet34Segmenter(Model):
    def __init__(self, model_path=None, device="cuda"):
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

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Modelo cargado correctamente desde {model_path}")
        else:
            print("ℹ Modelo inicializado con pesos preentrenados (ImageNet).")

    def getModel(self):
      return self.model


    def save(self, path="weights/unet_resnet34_transfer.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✅ Pesos del modelo guardados en: {path}")
