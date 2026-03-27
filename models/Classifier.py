import torch.nn as nn
import torch
import torchmetrics
import lightning as L

class Classifier(L.LightningModule):
    def __init__(self, model, threshold=0.5): # Volvemos a 0.5 para un entrenamiento equilibrado
        super().__init__()
        self.model = model
        
        # 1️⃣ Loss con Label Smoothing para mejorar la generalización (evita overfitting)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.threshold = threshold
        self.prepare_data_per_node = False

        # Métricas actualizadas
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc   = torchmetrics.Accuracy(task='binary')
        
        # Métricas de Test
        self.test_acc       = torchmetrics.Accuracy(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall    = torchmetrics.Recall(task='binary')
        self.test_f1        = torchmetrics.F1Score(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Cálculo de predicciones basado en el umbral
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > self.threshold).long()
        
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > self.threshold).long()
        
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > self.threshold).long()
        
        # Registro de métricas de test
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc(preds, y), prog_bar=True)
        self.log("test_precision", self.test_precision(preds, y), prog_bar=True)
        self.log("test_recall", self.test_recall(preds, y), prog_bar=True)
        self.log("test_f1", self.test_f1(preds, y), prog_bar=True)

    def configure_optimizers(self):
        # 2️⃣ Learning Rate Diferencial: LR más bajo para la base, más alto para la cabeza
        # Esto asume que tu modelo tiene los atributos .features y .classifier (como MobileNet/EfficientNet)
        optimizer = torch.optim.AdamW([
            {'params': self.model.features.parameters(), 'lr': 1e-5},
            {'params': self.model.attention.parameters(), 'lr': 1e-4}, 
            {'params': self.model.classifier.parameters(), 'lr': 1e-4}
        ], weight_decay=0.05) # 3️⃣ Weight decay aumentado para regularizar

        # 4️⃣ Scheduler más sensible para "limpiar" el ruido de la gráfica de validación
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            patience=2,  # Reducimos la paciencia para que sea más reactivo
            factor=0.1   # Reducción más drástica para estabilizar picos
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_loss"
            }
        }