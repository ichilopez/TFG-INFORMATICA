from abc import abstractmethod
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, trainloader, validation_loader, epochs, learning_rate,
              patience=7, min_epochs=10, delta=1e-3, device="cuda"):
        pass
    
    @abstractmethod
    def evaluate(self, testloader,metrics_path, device, output_path):
       pass

    @abstractmethod
    def predict(self, image_paths):
        pass

    @abstractmethod
    def save(self, path):
        pass
 