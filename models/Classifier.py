import torch
import torch.nn as nn
import torchmetrics
import lightning as L


class Classifier(L.LightningModule):
    def __init__(
        self,
        model,
        threshold=0.5,
        lr=1e-4,
        class_weights=(1.0, 2.0),
        label_smoothing=0.1
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.threshold = threshold
        self.prepare_data_per_node = False

        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float),
            label_smoothing=label_smoothing
        )

        # Métricas de train/val
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

        # Métricas de test
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")

        # Para guardar métricas del threshold ajustado
        self.tuned_val_metrics = None

    def forward(self, x, meta=None):
        return self.model(x, meta)

    def _unpack_batch(self, batch):
        # Pipeline unificado: siempre esperamos x, meta, y
        x, meta, y = batch
        return x, meta, y

    def _shared_step(self, batch):
        x, meta, y = self._unpack_batch(batch)

        logits = self(x, meta)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= self.threshold).long()

        return loss, probs, preds, y

    def training_step(self, batch, batch_idx):
        loss, probs, preds, y = self._shared_step(batch)

        self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, preds, y = self._shared_step(batch)

        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, probs, preds, y = self._shared_step(batch)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc(preds, y), prog_bar=True)
        self.log("test_precision", self.test_precision(preds, y), prog_bar=True)
        self.log("test_recall", self.test_recall(preds, y), prog_bar=True)
        self.log("test_f1", self.test_f1(preds, y), prog_bar=True)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=0.05
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=2,
            factor=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    @torch.no_grad()
    def tune_threshold(self, dataloader, mode="f1", min_recall=0.70):
        """
        Ajusta el umbral usando validación.

        mode:
          - 'f1': maximiza F1
          - 'precision_at_recall': maximiza precisión con recall >= min_recall

        Además:
          - imprime el threshold escogido
          - imprime las métricas de validación con ese threshold
          - intenta registrarlas en el logger de Lightning si existe
        """
        self.eval()

        all_probs = []
        all_targets = []

        for batch in dataloader:
            x, meta, y = self._unpack_batch(batch)

            x = x.to(self.device)
            meta = meta.to(self.device)
            y = y.to(self.device)

            logits = self(x, meta)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.append(probs.detach().cpu())
            all_targets.append(y.detach().cpu())

        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)

        thresholds = torch.arange(0.05, 0.96, 0.01)

        best_metrics = None
        eps = 1e-12

        for thr in thresholds:
            preds = (all_probs >= thr).long()

            tp = ((preds == 1) & (all_targets == 1)).sum().item()
            tn = ((preds == 0) & (all_targets == 0)).sum().item()
            fp = ((preds == 1) & (all_targets == 0)).sum().item()
            fn = ((preds == 0) & (all_targets == 1)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

            current = {
                "threshold": float(thr.item()),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            if mode == "f1":
                if (
                    best_metrics is None
                    or current["f1"] > best_metrics["f1"] + eps
                    or (
                        abs(current["f1"] - best_metrics["f1"]) <= eps
                        and current["threshold"] > best_metrics["threshold"]
                    )
                ):
                    best_metrics = current

            elif mode == "precision_at_recall":
                if recall < min_recall:
                    continue

                if (
                    best_metrics is None
                    or current["precision"] > best_metrics["precision"] + eps
                    or (
                        abs(current["precision"] - best_metrics["precision"]) <= eps
                        and current["f1"] > best_metrics["f1"] + eps
                    )
                    or (
                        abs(current["precision"] - best_metrics["precision"]) <= eps
                        and abs(current["f1"] - best_metrics["f1"]) <= eps
                        and current["threshold"] > best_metrics["threshold"]
                    )
                ):
                    best_metrics = current
            else:
                raise ValueError("mode debe ser 'f1' o 'precision_at_recall'")

        if best_metrics is None:
            raise RuntimeError(
                f"Ningún threshold alcanzó recall >= {min_recall:.2f}"
            )

        self.threshold = best_metrics["threshold"]
        self.tuned_val_metrics = {
            "mode": mode,
            "threshold": best_metrics["threshold"],
            "val_accuracy_at_tuned_threshold": best_metrics["accuracy"],
            "val_precision_at_tuned_threshold": best_metrics["precision"],
            "val_recall_at_tuned_threshold": best_metrics["recall"],
            "val_f1_at_tuned_threshold": best_metrics["f1"],
        }

        print(
            f"[tune_threshold] mode={mode} | "
            f"threshold={self.threshold:.2f} | "
            f"val_acc={best_metrics['accuracy']:.4f} | "
            f"val_precision={best_metrics['precision']:.4f} | "
            f"val_recall={best_metrics['recall']:.4f} | "
            f"val_f1={best_metrics['f1']:.4f}"
        )

        if self.logger is not None:
            try:
                self.logger.log_metrics(
                    {
                        "tuned_threshold": self.threshold,
                        "val_acc_at_tuned_threshold": best_metrics["accuracy"],
                        "val_precision_at_tuned_threshold": best_metrics["precision"],
                        "val_recall_at_tuned_threshold": best_metrics["recall"],
                        "val_f1_at_tuned_threshold": best_metrics["f1"],
                    },
                    step=self.current_epoch
                )
            except Exception:
                pass

        return self.tuned_val_metrics