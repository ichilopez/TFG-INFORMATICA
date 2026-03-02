import torch.nn as nn
import torch
import torchmetrics
import lightning as L

class Classifier(L.LightningModule):
    def __init__(self, model, threshold=0.34):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.threshold = threshold  # <-- Umbral configurable
        self.prepare_data_per_node = False

        # Métricas
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc   = torchmetrics.Accuracy(task='binary')
        self.test_acc  = torchmetrics.Accuracy(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall    = torchmetrics.Recall(task='binary')
        self.test_f1        = torchmetrics.F1Score(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:,1]
        preds = (probs > self.threshold).long()  # <-- usar la variable
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:,1]
        preds = (probs > self.threshold).long()  # <-- usar la variable
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:,1]
        preds = (probs > self.threshold).long()  # <-- usar la variable
        acc = self.test_acc(preds, y)
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)
        f1 = self.test_f1(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}