import yaml
import torch
import os
from utils.dataset import get_dataloaders
from utils.train import train_model, evaluate_model

def main(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(
        cfg["data"]["train_dir"],
        cfg["data"]["test_dir"],
        cfg["train"]["batch_size"],
        cfg["data"]["num_workers"],
    )

    model = SimpleCNN(num_classes=cfg["model"]["num_classes"])

    model = train_model(model, train_loader, device, cfg["train"]["epochs"], cfg["train"]["learning_rate"])
    acc = evaluate_model(model, test_loader, device)

    os.makedirs(cfg["save"]["output_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["save"]["output_dir"], cfg["save"]["model_name"]))
    print("Modelo guardado exitosamente ✅")

if __name__ == "__main__":
    main()
