import yaml
from models.EfficientNetB0Classifier import EfficientNetB0Classifier
from models.MobileNetV2Classifier import MobileNetV2Classifier
from models.ResNet18Classifier import ResNet18Classifier
from models.UNetResNet34Segmenter import UNetResNet34Segmenter
from  models.ViTSmallClassifier import ViTSmallClassifier
from  models.Classifier import Classifier
from  models.Segmenter import Segmenter
from utils.classification.ClassificationImageManager import ClassificationImageManager
from utils.segmentation.SegmentationImageManager import SegmentationImageManager
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning as L
from ultralytics import YOLO

def main(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    
    model_manager,image_manager = get_model(
    model_name=cfg["model"]["model_name"],
    batch_size = cfg["train"]["batch_size"],
    num_workers = cfg["train"]["num_workers"],
    num_classes=cfg["model"]["num_classes"]
    )

    early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    mode="min",
    verbose=True)

    model_path = cfg["model"]["model_path"]

    checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="best-{epoch:02d}-{val_loss:.4f}",
    dirpath= model_path
    )

    name_csv = cfg["model"]["model_name"]

    csv_logger = CSVLogger(
    save_dir="logs",
    name=name_csv
    )

    trainer = L.Trainer(
     max_epochs=30,
     accelerator="auto",
     devices="auto",
     callbacks=[early_stop, checkpoint],
     logger=csv_logger,
     log_every_n_steps=10)
    
    print("Training..")
    
    trainer.fit(model_manager,datamodule=image_manager)

    print("Testing...")
    trainer.test(model_manager,datamodule=image_manager)
    best_model_path = checkpoint.best_model_path
    print(f"✅ Mejor modelo guardado en: {best_model_path}")







def get_model(model_name: str, batch_size,num_workers,num_classes=2, model_path: str = None):

    model_name = model_name.lower()

    if model_name == 'mobilenetv2':
        model= MobileNetV2Classifier().getModel()
        image_manager = ClassificationImageManager(batch_size,num_workers)
        model_manager = Classifier(model)
        
    elif model_name == 'efficientnetb0':
        model = EfficientNetB0Classifier().getModel()
        image_manager = ClassificationImageManager(batch_size,num_workers)
        model_manager = Classifier(model)

    elif model_name == 'resnet18':
        model = ResNet18Classifier().getModel()
        image_manager = ClassificationImageManager(batch_size,num_workers)
        model_manager = Classifier(model)

    elif model_name == 'vitsmall':
        model = ViTSmallClassifier().getModel()
        image_manager = ClassificationImageManager(batch_size,num_workers)
        model_manager = Classifier(model)
    
    elif model_name == 'unetresnet34':
        model = UNetResNet34Segmenter().getModel()
        image_manager = SegmentationImageManager(batch_size,num_workers)
        model_manager = Segmenter(model)

    else:
        raise ValueError(f"Modelo no reconocido: '{model_name}'. Opciones válidas: "
                         f"'mobilenetv2', 'efficientnetb0', 'resnet18', 'vitsmall', 'yolo','unetresnet34'.")
    return model_manager,image_manager



def main_yolo():
    data_yaml = "configs/data.yaml"

    model = YOLO("yolov8n.pt")  

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)


    model.train(
        data=data_yaml,                       
        epochs=cfg["train"]["epochs"],       
        imgsz=640,                            
        batch=cfg["train"]["batch_size"],     
        patience=10,                          
        project="/content/drive/MyDrive/datos/yolo/runs",
        name="exp_yolov8_seg",
        exist_ok=True,
        freeze = [0] #congelo el encoder
    )

    print("✅ Entrenamiento completado. Ahora evaluamos sobre test set...")

    
    metrics = model.val(
        data=data_yaml,
        split="test",   
        imgsz=640,
        batch=16
    )

    print("✅ Evaluación completada. Métricas:")
    print(metrics)








if __name__ == "__main__":

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_name=cfg["model"]["model_name"]

    if(model_name=="yolo"): main_yolo()
    else: main()



