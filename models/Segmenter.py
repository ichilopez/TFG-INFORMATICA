import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import segmentation_models_pytorch as smp
import torchmetrics

class Segmenter(L.LightningModule):
    def __init__(self, 
                 model,
                 lr=1e-3,
                 num_classes=1):
        super().__init__()
        self.lr = lr
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()

        # métricas de segmentación 
        self.test_iou = torchmetrics.JaccardIndex(task="binary")
        self.test_dice = torchmetrics.F1Score(task="binary")  # Dice ~ F1 para segmentación

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)

        # ajustar tamaño de logits si difiere de mask
        if logits.shape[2:] != masks.shape[2:]:
            logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)

        if logits.shape[2:] != masks.shape[2:]:
            logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss = self.loss_fn(logits, masks)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        
        if logits.shape[2:] != masks.shape[2:]:
            logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

        # calcular loss
        loss = self.loss_fn(logits, masks)
        self.log("test_loss", loss, prog_bar=True)

        preds = torch.sigmoid(logits) > 0.5

        iou = self.test_iou(preds.int(), masks.int())
        dice = self.test_dice(preds.int(), masks.int())

        self.log("test_iou", iou, prog_bar=True)
        self.log("test_dice", dice, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)