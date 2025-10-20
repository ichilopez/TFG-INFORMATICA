import torch
import torch.nn as nn

def evaluate_model(model, input_loader, output_loader, device, study_type="classification"):
    model.eval()
    criterion_seg = nn.BCEWithLogitsLoss()  # para segmentación binaria
    criterion_clf = nn.CrossEntropyLoss()   # para clasificación

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (imgs, _), (targets, _) in zip(input_loader, output_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            if study_type == "segmentationStudy":
                # Asegurar que dimensiones coincidan
                if outputs.shape != targets.shape:
                    targets = targets.unsqueeze(1).float()  # [B,1,H,W]
                else:
                    targets = targets.float()
                loss = criterion_seg(outputs, targets)
                total_loss += loss.item()
            else:
                # Clasificación
                loss = criterion_clf(outputs, targets.long())
                total_loss += loss.item()

                # Cálculo de accuracy
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

    if study_type == "classification":
        acc = correct / total
        print(f"Accuracy: {acc:.4f} | Loss: {total_loss/len(input_loader):.4f}")
        return acc
    else:
        avg_loss = total_loss / len(input_loader)
        print(f"Segmentation Loss: {avg_loss:.4f}")
        return avg_loss
