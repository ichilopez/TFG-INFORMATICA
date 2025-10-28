
from utils.ImageManager import ImageManager
import os
import pandas as pd
path_mass_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_test_set.csv'
path_mass_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_train_set.csv'

data_extension = "1.jpeg"
output_extension = "2.jpeg"
root = 'C:\\Users\\Itziar\\Documents\\Documentos\\TFG-INF-DATOS\\archive\\jpeg'

class ClassificationImageManager(ImageManager):
    def getPaths(self): 
        train_data = pd.read_csv(path_mass_train)
        test_data = pd.read_csv(path_mass_test)
        train_input = self.__getPathsList(train_data)
        train_output = self.__getLabels(train_data)
        test_input = self.__getPathsList(test_data)
        test_output = self.__getLabels(test_data)

        return train_input, train_output, test_input, test_output
    
    def __getPathsList(self, data):
        data_list = []
        for i in range(len(data)):
            aux_list = data.loc[i, 'ROI mask file path'].split('/')
            folder_name = os.path.join(root, aux_list[-2])
            data_list.append(os.path.join(folder_name, data_extension)) 
        return data_list   
          
    def __getLabels(self, data):
        output = []
        for i in range(len(data)):
            output.append(data.loc[i, 'pathology'])
        return output
