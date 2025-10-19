
from abc import abstractmethod
root = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/jpeg'
class ImageManager:
    @abstractmethod
    def charge_df(self):
        pass

    @abstractmethod
    def getImagesPaths(self,test_data,train_data):
        pass

    @abstractmethod
    def getPathsList(self,data):
        pass

