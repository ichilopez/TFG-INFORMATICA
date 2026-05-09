import os
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


def save_model_with_threshold(model_manager, save_path, threshold=None, extra_metrics=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    payload = {
        "state_dict": model_manager.state_dict(),
        "threshold": threshold if threshold is not None else getattr(model_manager, "threshold", None),
        "extra_metrics": extra_metrics
    }
    torch.save(payload, save_path)


def main(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_manager, image_manager = get_model(
        model_name=cfg["model"]["model_name"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        num_classes=cfg["model"]["num_classes"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    model_path = cfg["model"]["model_path"]
    os.makedirs(model_path, exist_ok=True)

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        dirpath=model_path
    )

    name_csv = cfg["model"]["model_name"]

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

    # Cargar mejor checkpoint
    best_path = checkpoint.best_model_path
    ckpt = torch.load(best_path, map_location="cpu")
    model_manager.load_state_dict(ckpt["state_dict"])

    print(f"Mejor checkpoint cargado: {best_path}")

    # =========================
    # 1) Evaluación sin ajustar threshold
    # =========================
    print("\n=== TEST SIN AJUSTAR THRESHOLD ===")
    model_manager.threshold = 0.5
    test_results_fixed = trainer.test(model_manager, datamodule=image_manager)

    fixed_save_path = os.path.join(model_path, "model_threshold_fixed.pt")
    save_model_with_threshold(
        model_manager,
        fixed_save_path,
        threshold=model_manager.threshold,
        extra_metrics={"test_results": test_results_fixed}
    )
    print(f"Modelo con threshold fijo guardado en: {fixed_save_path}")

    # =========================
    # 2) Evaluación con threshold ajustado
    # =========================
    print("\n=== AJUSTE DE THRESHOLD EN VALIDACIÓN ===")
    best_metrics = model_manager.tune_threshold(
        image_manager.val_dataloader(),
        mode="precision_at_recall",
        min_recall=0.70
    )

    print(f"Threshold ajustado: {model_manager.threshold}")
    print(f"Métricas de validación con threshold ajustado: {best_metrics}")

    print("\n=== TEST CON THRESHOLD AJUSTADO ===")
    test_results_tuned = trainer.test(model_manager, datamodule=image_manager)

    tuned_save_path = os.path.join(model_path, "model_threshold_tuned.pt")
    save_model_with_threshold(
        model_manager,
        tuned_save_path,
        threshold=model_manager.threshold,
        extra_metrics={
            "val_threshold_metrics": best_metrics,
            "test_results": test_results_tuned
        }
    )
    print(f"Modelo con threshold ajustado guardado en: {tuned_save_path}")


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