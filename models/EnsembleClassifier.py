import torch


class EnsembleClassifier:
    def __init__(self, models, weights=None, threshold=0.5, device=None):
        self.models = models
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("weights y models deben tener la misma longitud")
            s = sum(weights)
            self.weights = [w / s for w in weights]

        for model in self.models:
            model.to(self.device)
            model.eval()

    @torch.no_grad()
    def predict_proba(self, x, meta):
        probs = []

        for model in self.models:
            logits = model(x, meta)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p)

        probs = torch.stack(probs, dim=0)  # [n_models, batch]
        weights = torch.tensor(self.weights, device=probs.device).unsqueeze(1)
        ensemble_probs = (probs * weights).sum(dim=0)

        return ensemble_probs

    @torch.no_grad()
    def predict(self, x, meta, threshold=None):
        thr = self.threshold if threshold is None else threshold
        probs = self.predict_proba(x, meta)
        return (probs >= thr).long()

    @torch.no_grad()
    def tune_threshold(self, dataloader, mode="f1", min_recall=0.70):
        all_probs = []
        all_targets = []

        for x, meta, y in dataloader:
            x = x.to(self.device)
            meta = meta.to(self.device)
            y = y.to(self.device)

            probs = self.predict_proba(x, meta)

            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

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
            raise RuntimeError("No se pudo ajustar threshold")

        self.threshold = best_metrics["threshold"]

        print(
            f"[ensemble tune_threshold] threshold={self.threshold:.2f} | "
            f"val_acc={best_metrics['accuracy']:.4f} | "
            f"val_precision={best_metrics['precision']:.4f} | "
            f"val_recall={best_metrics['recall']:.4f} | "
            f"val_f1={best_metrics['f1']:.4f}"
        )

        return best_metrics