import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import os
import numpy as np
import cv2
from tqdm import tqdm
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
        print("🧊 Encoder congelado (Transfer Learning puro activado).")

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Modelo cargado correctamente desde {model_path}")
        else:
            print("ℹ️ Modelo inicializado con pesos preentrenados (ImageNet).")


    def train(self, trainloader, epochs, learning_rate, device="cuda"):
        self.model.train()
        criterion = smp.losses.DiceLoss(mode='binary')
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),  # solo entrena el decoder
            lr=learning_rate
        )

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in tqdm(trainloader, desc=f"Época {epoch+1}/{epochs}"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"📉 Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(trainloader):.4f}")

        print("✅ Entrenamiento finalizado (solo decoder entrenado).")


    def evaluate(self, testloader, device="cuda"):
        self.model.eval()
        dice_score = 0.0
        with torch.no_grad():
            for batch in tqdm(testloader, desc="Evaluando"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = torch.sigmoid(self.model(images))
                preds = (outputs > 0.5).float()

                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice_score += (2. * intersection / (union + 1e-6)).item()

        dice_score /= len(testloader)
        print(f"✅ Dice score promedio: {dice_score:.4f}")
        return dice_score


    def predict(self, image_paths, threshold=0.5):
        self.model.eval()
        results = []

        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (256, 256))
            tensor = torch.tensor(img_resized / 255.0).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                pred_mask = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
                mask_bin = (pred_mask > threshold).astype(np.uint8)

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                bboxes.append({
                    "bbox": [x, y, x + w, y + h],
                    "score": float(pred_mask[y:y+h, x:x+w].mean())
                })

            results.append({
                "image_path": path,
                "bboxes": bboxes,
                "mask": mask_bin
            })

        return results


    def save(self, path="weights/unet_resnet34_transfer.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✅ Pesos del modelo guardados en: {path}")
