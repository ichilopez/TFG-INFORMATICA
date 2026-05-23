import yaml
import torch
import lightning as L

from models.EfficientNetB0Classifier import EfficientNetB0Classifier
from models.EfficientNetB0ClassifierMultimodal import EfficientNetB0ClassifierMultimodal
from models.MobileNetV2Classifier import MobileNetV2Classifier
from models.MobileNetV2ClassifierMultimodal import MobileNetV2ClassifierMultimodal
from models.ResNet18Classifier import ResNet18Classifier
from models.ResNet18ClassifierMultimodal import ResNet18ClassifierMultimodal
from models.UNetResNet34Segmenter import UNetResNet34Segmenter
from models.ViTSmallClassifier import ViTSmallClassifier
from models.ViTSmallClassifierMultimodal import ViTSmallClassifierMultimodal
from models.Classifier import Classifier
from models.Segmenter import Segmenter

from utils.classification.ClassificationImageManager import ClassificationImageManager
from utils.segmentation.SegmentationImageManager import SegmentationImageManager

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def main(config_path="configs/config.yaml"):
    # Carga la configuración del experimento
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Crea el modelo y el gestor de datos según el YAML
    model_manager, image_manager = get_model(
        model_name=cfg["model"]["model_name"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        num_classes=cfg["model"]["num_classes"]
    )

    # Detiene el entrenamiento si val_loss deja de mejorar
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    model_path = cfg["model"]["model_path"]

    # Guarda el mejor modelo según val_loss
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        dirpath=model_path
    )

    name_csv = cfg["model"]["model_name"]

    # Guarda las métricas en archivos CSV
    csv_logger = CSVLogger(
        save_dir="logs",
        name=name_csv
    )

    trainer = L.Trainer(
        max_epochs=cfg["train"]["epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, checkpoint],
        logger=csv_logger,
        log_every_n_steps=10
    )

    print("Training...")
    trainer.fit(model_manager, datamodule=image_manager)

    # Carga el mejor checkpoint antes de evaluar
    best_path = checkpoint.best_model_path
    ckpt = torch.load(best_path, map_location="cpu")
    model_manager.load_state_dict(ckpt["state_dict"])

    print(f"Mejor checkpoint cargado: {best_path}")

    print("\n=== TEST CON UMBRAL FIJO (0.5) ===")

    # Evaluación con umbral estándar
    model_manager.threshold = 0.5
    trainer.test(model_manager, datamodule=image_manager)

    print("\n=== AJUSTE DE UMBRAL EN VALIDACIÓN ===")

    # Ajuste del umbral usando validación
    best_metrics = model_manager.tune_threshold(
        image_manager.val_dataloader(),
        mode="precision_at_recall",
        min_recall=0.70
    )

    print(f"Threshold ajustado: {model_manager.threshold}")
    print(f"Métricas de validación: {best_metrics}")

    print("\n=== TEST CON UMBRAL AJUSTADO ===")

    # Evaluación final con el umbral ajustado
    trainer.test(model_manager, datamodule=image_manager)


def get_model(model_name: str, batch_size, num_workers, num_classes=2):
    model_name = model_name.lower()

    if model_name == "mobilenetv2":
        model = MobileNetV2Classifier().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "mobilenetv2_multi":
        model = MobileNetV2ClassifierMultimodal().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "efficientnetb0":
        model = EfficientNetB0Classifier().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "efficientnetb0_multi":
        model = EfficientNetB0ClassifierMultimodal().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "resnet18":
        model = ResNet18Classifier().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "resnet18_multi":
        model = ResNet18ClassifierMultimodal().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "vitsmall":
        model = ViTSmallClassifier().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "vitsmall_multi":
        model = ViTSmallClassifierMultimodal().getModel()
        image_manager = ClassificationImageManager(batch_size, num_workers)
        model_manager = Classifier(model)

    elif model_name == "unetresnet34":
        # Modelo de segmentación, no de clasificación
        model = UNetResNet34Segmenter().getModel()
        image_manager = SegmentationImageManager(batch_size, num_workers)
        model_manager = Segmenter(model)

    else:
        raise ValueError(
            f"Modelo no reconocido: '{model_name}'. Opciones válidas: "
            f"'mobilenetv2', 'mobilenetv2_multi', 'efficientnetb0', "
            f"'efficientnetb0_multi', 'resnet18', 'resnet18_multi', "
            f"'vitsmall', 'vitsmall_multi', 'unetresnet34'."
        )

    return model_manager, image_manager


if __name__ == "__main__":
    main()