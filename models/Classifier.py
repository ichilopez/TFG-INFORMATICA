import torch.nn as nn
import torch.optim as optim
from models.Model import Model
import torch
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import yaml

class Classifier(Model):
    def __init__(self):
        super().__init__()

    def train(self, trainloader, validationloader, epochs=20, learning_rate=1e-4,
              device="cuda", patience=5, delta=0.001):

        self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=learning_rate
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for inputs, labels,img_path in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(trainloader)


            val_loss = self.validate(validationloader, criterion, device)

            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss - delta:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Mejora detectada (Val Loss ↓ {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"Sin mejora ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print("Early stopping activado. Entrenamiento detenido.")
                    break

    def validate(self, dataloader, criterion, device):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels,img_path in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, testloader, metrics_path="results/metrics.csv",
                 device="cuda"):
        with open("configs/config.yaml","r") as f:
            cfg = yaml.safe_load(f)
        
        output_path = cfg["save"]["results_path"]
        metrics_path = cfg["save"]["metrics_path"]

        self.model.to(device)
        self.model.eval()

        all_results = []
        running_loss = 0.0
        y_true_all, y_pred_all = [], []

        criterion = nn.CrossEntropyLoss()
        label_map_inv = {0: "BENIGN", 1: "MALIGNANT"}

        with torch.no_grad():
            for batch in testloader:
                inputs, labels, image_paths= batch
                image_paths = [None] * len(inputs)

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1) #obtenemos la clase con mayor probabilidad
                y_true_all.extend(labels.cpu().numpy()) #
                y_pred_all.extend(predicted.cpu().numpy())

                for img_path, true, pred in zip(image_paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                    all_results.append({
                        "image file path": img_path,
                        "true label": label_map_inv.get(int(true), "UNKNOWN"),
                        "predicted label": label_map_inv.get(int(pred), "UNKNOWN")
                    })

        avg_loss = running_loss / len(testloader)
        accuracy = (torch.tensor(y_true_all) == torch.tensor(y_pred_all)).float().mean().item()
        precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(all_results).to_csv(output_path, index=False)

        metrics_dict = {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)

        print(f"Resultados guardados en: {output_path}")
        print(f"Métricas globales guardadas en: {metrics_path}")
        print(
            f"🔹 Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

