import os
import torch
import pandas as pd
from ultralytics import YOLO
from torchvision.ops import box_iou
from models.Model import Model
import numpy as np
import yaml
import tempfile
from PIL import Image

class YOLOSegmenter(Model):
    def __init__(self, model_path=None):
        super().__init__()
        if model_path:
            self.model = YOLO(model_path)
            self.model.freeze()
            for name, param in self.model.named_parameters():
             if 'head' in name:  #solo descongelamos las últimas capas
              param.requires_grad = True 
        else:
            self.model = YOLO('yolov8n.pt')


    def _create_yolo_dataset_structure(self, data_info, base_dir, subset="train"):
     images_dir = os.path.join(base_dir, "images", subset)
     labels_dir = os.path.join(base_dir, "labels", subset)

     os.makedirs(images_dir, exist_ok=True)
     os.makedirs(labels_dir, exist_ok=True)

     for sample in data_info:
        img_path = sample["file_path_image"]
        img_name = os.path.basename(img_path)
        new_img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(new_img_path):
            try:
                os.symlink(os.path.abspath(img_path), new_img_path)
            except OSError:
                from shutil import copy2
                copy2(img_path, new_img_path)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            img = Image.open(img_path)
            w, h = img.size
            with open(label_path, "w") as f:
                for bbox in sample["bboxes"]:
                    x_min, y_min, x_max, y_max = bbox
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

     return images_dir, labels_dir

  
    def train(self, trainLoader, valLoader, epochs=50, learning_rate=1e-3, device="cuda", freeze_layers=10):
        
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Creando dataset temporal para YOLO en: {tmpdir}")

            train_images_dir, _ = self._create_yolo_dataset_structure(
                trainLoader.dataset.data_info, tmpdir, subset="train"
            )

            val_images_dir = self._create_yolo_dataset_structure(
                valLoader.dataset.data_info, tmpdir, subset="validation"
            )

            data_yaml_path = os.path.join(tmpdir, 'dataset.yaml')
            data_dict = {
                'train': train_images_dir,
                'val': val_images_dir,
                'nc': 1,
                'names': ['lesion']
            }
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_dict, f)

            print("Iniciando entrenamiento YOLOv8 ...")
            self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=8,
                freeze=freeze_layers,
                device=device,
                lr0=learning_rate,
                patience = 10
            )


    def predict(self, image_paths, device="cuda", conf_threshold=0.25):
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

            if bbox_true.numel() > 0 and boxes_pred.numel() > 0:
                ious = box_iou(bbox_true, boxes_pred)
                iou = ious.max().item()
            else:
                iou = 0.0

            dice = (2 * iou) / (iou + 1e-6)
            precision = iou
            recall = iou
            specificity = 1 - (1 - iou)
            accuracy = iou
            f1 = dice

            metrics_global["iou"].append(iou)
            metrics_global["dice"].append(dice)
            metrics_global["precision"].append(precision)
            metrics_global["recall"].append(recall)
            metrics_global["specificity"].append(specificity)
            metrics_global["accuracy"].append(accuracy)
            metrics_global["f1"].append(f1)

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

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(results_path, index=False)

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
