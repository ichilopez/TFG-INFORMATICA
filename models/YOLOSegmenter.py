import os
import torch
import pandas as pd
from ultralytics import YOLO
from torchvision.ops import box_iou
from models.Model import Model
import numpy as np

class YOLOSegmenter(Model):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = YOLO(model_path if model_path else 'yolov8n.pt')

    def train(self, trainloader, epochs, learning_rate, device="cuda", freeze_layers=10):
        import yaml
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            train_images = [item['file_path_image'] for item in trainloader.dataset.data_info]
            val_images = train_images
            data_yaml_path = os.path.join(tmpdir, 'dataset.yaml')
            data_dict = {'train': train_images, 'val': val_images, 'nc': 1, 'names': ['lesion']}
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_dict, f)

            self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=8,
                freeze=freeze_layers,
                device=device,
                lr0=learning_rate
            )

    def predict(self, image_paths, device="cuda", conf_threshold=0.25):
        """
        Realiza predicciones y devuelve bboxes en el mismo formato que evaluate.
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
            if len(boxes_pred) == 0:
                boxes_pred = torch.zeros((0, 4), device=device)
            else:
                boxes_pred = torch.tensor(boxes_pred, device=device)
            results_all.append({"image_path": img_path, "boxes_pred": boxes_pred})
        return results_all

    def evaluate(self, testloader, results_path="results/results_yolo.csv",
                 metrics_path="results/metrics_yolo.csv", device="cuda"):
        """
        Evalúa el modelo YOLO usando predict() y calcula métricas comparables con UNet.
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

            # Calcular IoU máximo
            if bbox_true.numel() > 0 and boxes_pred.numel() > 0:
                ious = box_iou(bbox_true, boxes_pred)
                iou = ious.max().item()
            else:
                iou = 0.0

            # Métricas derivadas
            dice = (2 * iou) / (iou + 1e-6)
            precision = iou
            recall = iou
            specificity = 1 - (1 - iou)
            accuracy = iou
            f1 = dice

            # Guardar métricas por imagen
            metrics_global["iou"].append(iou)
            metrics_global["dice"].append(dice)
            metrics_global["precision"].append(precision)
            metrics_global["recall"].append(recall)
            metrics_global["specificity"].append(specificity)
            metrics_global["accuracy"].append(accuracy)
            metrics_global["f1"].append(f1)

            # Guardar resultados individuales
            all_results.append({
                "image_path": sample['file_path_image'],
                "bbox_true": bbox_true.cpu().numpy().tolist() if bbox_true.numel() > 0 else [],
                "bbox_pred": boxes_pred.cpu().numpy().tolist() if boxes_pred.numel() > 0 else [],
                "iou": float(iou),
                "dice": float(dice),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "accuracy": float(accuracy),
                "f1": float(f1)
            })

        # Guardar CSV detallado
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(results_path, index=False)

        # Métricas globales promedio
        mean_metrics = {f"{k}_mean": np.mean(v) for k, v in metrics_global.items()}
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        pd.DataFrame([mean_metrics]).to_csv(metrics_path, index=False)

        print(f"✅ Resultados guardados en: {results_path}")
        print(f"📊 Métricas promedio guardadas en: {metrics_path}")
        print(
            f"🔹 IoU={mean_metrics['iou_mean']:.4f}, Dice={mean_metrics['dice_mean']:.4f}, "
            f"Precision={mean_metrics['precision_mean']:.4f}, Recall={mean_metrics['recall_mean']:.4f}, "
            f"F1={mean_metrics['f1_mean']:.4f}, Accuracy={mean_metrics['accuracy_mean']:.4f}"
        )

        return mean_metrics, df_results

    def save(self, path="weights/yolo.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.model.save(path)
            print(f"✅ Modelo YOLO guardado correctamente en: {path}")
        except AttributeError:
            torch.save(self.model.model.state_dict(), path)
            print(f"✅ Pesos guardados manualmente en: {path}")
