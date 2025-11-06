from ultralytics import YOLO
import tempfile
import os
import yaml
from models.Model import Model
import torch

class YOLOSegmenter(Model):
    def __init__(self, model_path=None):
        super().__init__() 
        self.model = YOLO(model_path if model_path else 'yolov8n.pt')  

    def train(self, trainloader, epochs, learning_rate, device="cuda", freeze_layers=10):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_images = [item['file_path_image'] for item in trainloader.dataset.data_info]
            val_images = train_images  

            data_yaml_path = os.path.join(tmpdir, 'dataset.yaml')
            data_dict = {
                'train': train_images,
                'val': val_images,
                'nc': 1,
                'names': ['lesion']
            }
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

    def evaluate(self, testloader, device="cuda"):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_images = [item['file_path_image'] for item in testloader.dataset.data_info]
            data_yaml_path = os.path.join(tmpdir, 'dataset_val.yaml')
            data_dict = {
                'val': test_images,
                'nc': 1,
                'names': ['lesion']
            }
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_dict, f)

            results = self.model.val(data=data_yaml_path, device=device)
            return results

    def predict(self, image_paths, conf_threshold=0.25, save_visuals=False, output_dir="predictions"):
        os.makedirs(output_dir, exist_ok=True)
        results = self.model(image_paths, conf=conf_threshold, verbose=False)

        parsed_results = []
        for img_path, result in zip(image_paths, results):
            boxes_info = []

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item()) if box.cls is not None else 0

                    boxes_info.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls,
                        "label": self.model.names.get(cls, "lesion")
                    })

            parsed_results.append({
                "image_path": img_path,
                "boxes": boxes_info
            })

        return parsed_results
    
    def save(self, path="weights/yolo.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.model.save(path)
            print(f"✅ Modelo YOLO guardado correctamente en: {path}")
        except AttributeError:
            torch.save(self.model.model.state_dict(), path)
            print(f"✅ Pesos guardados manualmente en: {path}")

