import torch.nn as nn
import torch
import torchmetrics
from sklearn.metrics import precision_score, recall_score, f1_score
import lightning as L

class Classifier(L.LightningModule):
    def __init__(self,model,lr=1e-3):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model
        self.lr=lr
         # Definir métricas con torchmetrics
        self.test_accuracy = torchmetrics.Accuracy(task="binary",threshold=0.5)
        self.test_precision = torchmetrics.Precision(num_classes=2, average='macro',task="binary",threshold=0.5)
        self.test_recall = torchmetrics.Recall(num_classes=2, average='macro',task="binary",threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(num_classes=2, average='macro',task="binary",threshold=0.5)
        
    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits,y)
        self.log("train_loss",loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),self.lr)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        # registramos loss
        self.log("test_loss", loss, prog_bar=True)

        # calculamos métricas
        acc = self.test_accuracy(preds, y)
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)
        f1 = self.test_f1(preds, y)

        # registramos métricas
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)



