import torch.nn as nn
import torch
import torchmetrics
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import lightning as L

class Classifier(L.LightningModule):
    def __init__(self,model,lr=1e-3):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model
        self.lr=lr
        self.test_accuracy = torchmetrics.Accuracy(task ='binary',num_classes=2)
        self.test_precision = torchmetrics.Precision(num_classes=2,task ='binary',average='macro')
        self.test_recall = torchmetrics.Recall(num_classes=2, task ='binary', average='macro')
        self.test_f1 = torchmetrics.F1Score(num_classes=2, task ='binary', average='macro')
        self.prepare_data_per_node = False 

        
    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits,y)
        self.log("train_loss",loss)
        return loss
    
    

    def configure_optimizers(self):

     backbone_params = []
     classifier_params = []

     for name, param in self.model.named_parameters():
         if param.requires_grad:
             if "features" in name:
                 backbone_params.append(param)
             else:
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



