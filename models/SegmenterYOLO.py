from ultralytics import YOLO
import tempfile
import os
import yaml
from models.Model import Model

class SegmenterYOLO(Model):
    def __init__(self, model_path=None):
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')  

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

            # Entrenamiento YOLO
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

    def predict(self, image_paths):
        results = self.model(image_paths)
        return results
