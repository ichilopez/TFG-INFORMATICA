import os
import torch
import pandas as pd
from ultralytics import YOLO
from torchvision.ops import box_iou
from models.Model import Model
import numpy as np
import yaml

class YOLOSegmenter(Model):
    def __init__(self, model_path=None):
        super().__init__()
        # Cargar modelo
        self.model = YOLO(model_path if model_path else "yolov8n.pt")
        if model_path:
            self.model.freeze()
        for name, param in self.model.named_parameters():
                if 'head' in name:
                    param.requires_grad = True

    def train(self, epochs=50, learning_rate=1e-3, device="cuda", freeze_layers=10, batch_size=8):
        """
        Entrena YOLOv8 usando un archivo data.yaml con rutas absolutas a las imágenes.
        Los .txt deben estar generados junto a cada imagen.
        """
        with open("configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        data_yaml_path = cfg["data"]["data_yaml_path"]
        print(f"Iniciando entrenamiento YOLOv8 con {epochs} epochs...")
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            freeze=freeze_layers,
            device=device,
            lr0=learning_rate,
            patience=10
        )
        print("✅ Entrenamiento finalizado")

    def predict(self, image_paths, device="cuda", conf_threshold=0.25):
        """
        Realiza predicciones sobre una o varias imágenes.
        """
        self.model.to(device)
        self.model.eval()
        results_all = []

        for img_path in image_paths:
            results = self.model(img_path, conf=conf_threshold, verbose=False, device=device)
            boxes_pred = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        boxes_pred.append([x1, y1, x2, y2])
            boxes_pred = torch.tensor(boxes_pred, device=device) if boxes_pred else torch.zeros((0, 4), device=device)
            results_all.append({"image_path": img_path, "boxes_pred": boxes_pred})
        return results_all

    def evaluate(self, testloader, results_path="results/results_yolo.csv",
                 metrics_path="results/metrics_yolo.csv", device="cuda"):
        """
        Evalúa el modelo usando bounding boxes verdaderas en testloader.dataset.data_info.
        """
        self.model.to(device)
        self.model.eval()

        image_paths = [item['file_path_image'] for item in testloader.dataset.data_info]
        predictions = self.predict(image_paths, device=device)

        all_results = []
        metrics_global = {k: [] for k in ["iou", "dice", "precision", "recall", "specificity", "accuracy", "f1"]}

        for sample, pred in zip(testloader.dataset.data_info, predictions):
            bbox_true = torch.tensor([[float(sample['x_min']), float(sample['y_min']),
                                       float(sample['x_max']), float(sample['y_max'])]], device=device)
            boxes_pred = pred["boxes_pred"]

            iou = box_iou(bbox_true, boxes_pred).max().item() if bbox_true.numel() > 0 and boxes_pred.numel() > 0 else 0.0
            dice = (2 * iou) / (iou + 1e-6)
            precision = iou
            recall = iou
            specificity = 1 - (1 - iou)
            accuracy = iou
            f1 = dice

            for k, v in zip(metrics_global.keys(), [iou, dice, precision, recall, specificity, accuracy, f1]):
                metrics_global[k].append(v)

            all_results.append({
                "image_path": sample['file_path_image'],
                "bbox_true": bbox_true.cpu().numpy().tolist() if bbox_true.numel() > 0 else [],
                "bbox_pred": boxes_pred.cpu().numpy().tolist() if boxes_pred.numel() > 0 else [],
                "iou": iou, "dice": dice, "precision": precision, "recall": recall,
                "specificity": specificity, "accuracy": accuracy, "f1": f1
            })

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        pd.DataFrame(all_results).to_csv(results_path, index=False)

        mean_metrics = {f"{k}_mean": np.mean(v) for k, v in metrics_global.items()}
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        pd.DataFrame([mean_metrics]).to_csv(metrics_path, index=False)

        print(f"✅ Resultados guardados en: {results_path}")
        print(f"📊 Métricas promedio guardadas en: {metrics_path}")
        return mean_metrics, pd.DataFrame(all_results)

    def save(self, path="weights/yolo.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.model.save(path)
            print(f"✅ Modelo YOLO guardado correctamente en: {path}")
        except AttributeError:
            torch.save(self.model.model.state_dict(), path)
            print(f"✅ Pesos guardados manualmente en: {path}")
