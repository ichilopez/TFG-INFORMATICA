import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import os
import numpy as np
import cv2
from tqdm import tqdm
from models.Model import Model
from torchvision.ops import masks_to_boxes
import pandas as pd

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
        print("Encoder congelado (Transfer Learning puro activado).")

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Modelo cargado correctamente desde {model_path}")
        else:
            print("ℹ Modelo inicializado con pesos preentrenados (ImageNet).")


    def train(self, trainloader, validation_loader, epochs, learning_rate,
              patience=7, min_epochs=10, delta=1e-3, device="cuda"):

        self.model.to(device)
        criterion = smp.losses.DiceLoss(mode='binary')
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0

        print(f" Entrenando durante {epochs} épocas con early stopping (patience={patience})...\n")

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(trainloader, desc=f"Época {epoch}/{epochs}"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(trainloader)
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in validation_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(validation_loader)

            print(f"📉 Epoch [{epoch}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss + delta < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch >= min_epochs and epochs_no_improve >= patience:
                print(f"⏹️ Early stopping activado en la época {epoch}. "
                      f"No hay mejora en {patience} épocas consecutivas.")
                break

        print("✅ Entrenamiento finalizado (solo decoder entrenado).")

    def evaluate(self, dataloader, output_csv, metrics_path, threshold=0.5):
        self.eval()
        self.to(self.device)

        all_results = []
        metrics_global = {
            "iou": [], "dice": [],
            "precision": [], "recall": [],
            "specificity": [], "accuracy": [], "f1": []
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluando UNet"):
                imgs = batch["image"].to(self.device)
                true_masks = batch["mask"].to(self.device)
                image_paths = batch["image_path"]

                preds = torch.sigmoid(self(imgs))
                pred_bin = (preds > threshold).to(torch.uint8)
                true_bin = (true_masks > 0.5).to(torch.uint8)

                for i in range(len(image_paths)):
                    pred_np = pred_bin[i].cpu().numpy().squeeze()
                    true_np = true_bin[i].cpu().numpy().squeeze()

                    TP = np.logical_and(pred_np == 1, true_np == 1).sum()
                    TN = np.logical_and(pred_np == 0, true_np == 0).sum()
                    FP = np.logical_and(pred_np == 1, true_np == 0).sum()
                    FN = np.logical_and(pred_np == 0, true_np == 1).sum()

                    precision = TP / (TP + FP + 1e-8)
                    recall = TP / (TP + FN + 1e-8)
                    specificity = TN / (TN + FP + 1e-8)
                    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

                    intersection = TP
                    union = TP + FP + FN
                    iou = intersection / (union + 1e-8)
                    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

                    for k, v in zip(
                        ["iou", "dice", "precision", "recall", "specificity", "accuracy", "f1"],
                        [iou, dice, precision, recall, specificity, accuracy, f1],
                    ):
                        metrics_global[k].append(v)

                    boxes_pred = masks_to_boxes(pred_bin[i]) if pred_bin[i].sum() > 0 else torch.empty((0, 4))
                    boxes_true = masks_to_boxes(true_bin[i]) if true_bin[i].sum() > 0 else torch.empty((0, 4))

                    all_results.append({
                        "image_path": image_paths[i],
                        "bbox_true": boxes_true.cpu().numpy().tolist() if boxes_true.numel() > 0 else [],
                        "bbox_pred": boxes_pred.cpu().numpy().tolist() if boxes_pred.numel() > 0 else [],
                        "iou": float(iou),
                        "dice": float(dice),
                        "precision": float(precision),
                        "recall": float(recall),
                        "specificity": float(specificity),
                        "accuracy": float(accuracy),
                        "f1": float(f1)
                    })

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)

        mean_metrics = {k: np.mean(v) for k, v in metrics_global.items()}
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        pd.DataFrame([mean_metrics]).to_csv(metrics_path, index=False)

        print(f"✅ Resultados detallados guardados en: {output_csv}")
        print(f"📊 Métricas globales guardadas en: {metrics_path}")
        print(
            f"🔹 IoU={mean_metrics['iou']:.4f}, Dice={mean_metrics['dice']:.4f}, "
            f"Precision={mean_metrics['precision']:.4f}, Recall={mean_metrics['recall']:.4f}, "
            f"F1={mean_metrics['f1']:.4f}, Accuracy={mean_metrics['accuracy']:.4f}"
        )

        return mean_metrics, df

    def predict(self, image_paths, threshold=0.5):
        self.eval()
        self.to(self.device)
        predictions = []

        with torch.no_grad():
            for path in tqdm(image_paths, desc="Prediciendo UNet"):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (256, 256))
                img = np.stack([img] * 3, axis=-1)
                tensor = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

                pred_mask = torch.sigmoid(self.model(tensor))[0, 0].cpu()
                pred_bin = (pred_mask > threshold).to(torch.uint8)
                boxes_pred = masks_to_boxes(pred_bin) if pred_bin.sum() > 0 else torch.empty((0, 4))

                predictions.append({
                    "image_path": path,
                    "mask_pred": pred_bin.numpy(),
                    "bbox_pred": boxes_pred.cpu().numpy().tolist() if boxes_pred.numel() > 0 else []
                })

        return predictions

    def save(self, path="weights/unet_resnet34_transfer.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✅ Pesos del modelo guardados en: {path}")
