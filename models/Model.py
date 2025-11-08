from abc import abstractmethod
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, trainloader, epochs, learning_rate, device="cuda"):
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
 