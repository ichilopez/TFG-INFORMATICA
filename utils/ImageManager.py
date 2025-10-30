
from abc import abstractmethod
class ImageManager:
    @abstractmethod
    def getDataLoaders(self, batch_size, num_workers, train_transform=None, test_transform=None):
        pass


