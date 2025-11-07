
from abc import abstractmethod
class ImageManager:
    @abstractmethod
    def getDataLoaders(self, batch_size, num_workers,model_name):
        pass


