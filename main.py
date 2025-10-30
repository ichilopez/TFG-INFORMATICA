import yaml
import torch
import os
from utils import dataset 
from torchvision import models
def main(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_dataLoader, test_dataLoader = dataset.getDataLoaders(
        cfg["data"]["study_type"],
        cfg["save"]["model_name"],
        cfg["train"]["batch_size"],
        cfg["data"]["num_workers"]
    )
    
    model = getModel(cfg["save"]["model_name"])
    print("Training...")
    print("Evaluating...")
    # model.train(model, train_dataLoader, device, cfg["train"]["epochs"], cfg["train"]["learning_rate"],cfg["data"]["study_type"])
    # model.evaluate(model, test_dataLoader, device,cfg["data"]["study_type"])
    os.makedirs(cfg["save"]["output_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["save"]["output_dir"], cfg["save"]["model_name"]))
    print("Modelo guardado exitosamente")

def getModel (modelName):
    if modelName == "resnet34":
        return models.resnet18(pretrained=True)


if __name__ == "__main__":
    main()
