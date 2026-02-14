from abc import abstractmethod
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def getModel(self):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
 