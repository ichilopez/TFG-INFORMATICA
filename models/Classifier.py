import torch.nn as nn
import torch
import torchmetrics
import lightning as L
import torch
from models.MobileNetV2Classifier import MobileNetV2Classifier

class Classifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_precision = torchmetrics.Precision( task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')
        self.prepare_data_per_node = False 
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
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.34).long()
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "features" in name:
                    backbone_params.append(param)
                elif "classifier" in name:
                    classifier_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 1e-4},
                {"params": classifier_params, "lr": 1e-3},
            ],
            weight_decay=1e-4
        )

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.34).long()

        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.34).long()
        self.log("test_loss", loss, prog_bar=True)
        acc = self.test_accuracy(preds, y)
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)
        f1 = self.test_f1(preds, y)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)


    def load_model(ckpt_path, model_name, threshold=0.6, device='cpu'):
     backbone = MobileNetV2Classifier(num_classes=2, in_channels=3).getModel()
     model = Classifier.load_from_checkpoint(ckpt_path, model=backbone, threshold=threshold, map_location=device)
     model.eval()
     model.to(device)
     return model

    def predict_batch(model, images_tensor):
     """images_tensor: [batch, 3, 224, 224]"""
     device = next(model.parameters()).device
     images_tensor = images_tensor.to(device)
     with torch.no_grad():
        logits = model(images_tensor)
        probs = torch.softmax(logits, dim=1)[:,1]
        preds = (probs > model.threshold).long()
     return preds.cpu(), probs.cpu()