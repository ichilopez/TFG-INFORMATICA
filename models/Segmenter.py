import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics


class Segmenter(L.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        threshold=0.5
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.threshold = threshold

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_iou = torchmetrics.JaccardIndex(task="binary")
        self.val_dice = torchmetrics.F1Score(task="binary")

        self.test_iou = torchmetrics.JaccardIndex(task="binary")
        self.test_dice = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        imgs, masks = batch
        masks = masks.float()

        logits = self(imgs)

        if logits.shape[2:] != masks.shape[2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        loss = self.loss_fn(logits, masks)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        targets = masks.int()

        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)

        self.val_iou(preds, targets)
        self.val_dice(preds, targets)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_iou", self.val_iou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_dice", self.val_dice, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)

        self.test_iou(preds, targets)
        self.test_dice(preds, targets)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_iou", self.test_iou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_dice", self.test_dice, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=3,
            factor=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }