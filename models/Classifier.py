import torch.nn as nn
import torch.optim as optim
from models.Model import Model
import torch
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class Classifier(Model):
    def __init__(self):
        super().__init__()

    def train(self, trainloader, epochs, learning_rate, device="cuda"):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=learning_rate)
        for epoch in range(1, epochs + 1):
            self.train() 
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step() 
                running_loss += loss.item()
            avg_loss = running_loss / len(trainloader)
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, testloader, device="cuda", output_path="results/evaluation_results.csv"):
        self.eval()
        self.to(device)
        criterion = nn.CrossEntropyLoss()

        all_results = []
        running_loss = 0.0
        y_true_all, y_pred_all = [], []

        with torch.no_grad():
            for batch in testloader:
                if isinstance(batch, dict):
                    inputs, labels = batch["image"], batch["label"]
                    image_paths = batch.get("image_path", [None] * len(inputs))
                else:
                    inputs, labels = batch
                    image_paths = [None] * len(inputs)

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                confs, predicted = torch.max(probs, 1)

                y_true_all.extend(labels.cpu().numpy())
                y_pred_all.extend(predicted.cpu().numpy())

                # Guardar resultados individuales
                for img_path, true, pred, score in zip(image_paths, labels.cpu().numpy(), predicted.cpu().numpy(), confs.cpu().numpy()):
                    all_results.append({
                        "image file path": img_path,
                        "true label": int(true),
                        "predicted label": int(pred),
                        "pred_score": float(score)
                    })

        # --- Métricas globales ---
        avg_loss = running_loss / len(testloader)
        accuracy = (torch.tensor(y_true_all) == torch.tensor(y_pred_all)).float().mean().item()
        precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true_all, y_pred_all)

        # Guardar resultados individuales
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_path, index=False)

        # Guardar métricas globales
        metrics_path = output_path.replace(".csv", "_metrics.csv")
        metrics_dict = {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)

        print(f"✅ Resultados guardados en: {output_path}")
        print(f"📊 Métricas globales guardadas en: {metrics_path}")
        print(
            f"🔹 Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        return metrics_dict, df_results, conf_matrix
