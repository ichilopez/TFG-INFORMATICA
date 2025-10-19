
from ImageManager import ImageManager
import pandas as pd
import os
path_mass_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_test_set.csv'
path_mass_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_train_set.csv'
data_extension = "1.jpeg"
output_extension = "2.jpeg"
#1216

class SegmentationImageManager (ImageManager):
    def charge_df(self):
        # Cargar CSVs de masas
        mass_test_data = pd.read_csv(path_mass_test)
        mass_train_data = pd.read_csv(path_mass_train)
        return mass_test_data, mass_train_data      

    def getImagesPaths(self,test_data,train_data): 
     mass_train, mass_test = self.getPathsList(train_data)
     masks_train , masks_test = self.getPathsList(test_data)
     return mass_train, mass_test ,masks_train ,masks_test
    
    def getPathsList(self,data):
     data_list = []
     output_list = []
     for i in range(len(data)):
        aux_list = data.loc[i, 'ROI mask file path'].split('/')
        folder_name = os.path.join(ImageManager.root, aux_list[-2])
        data_list.append(os.path.join(folder_name, data_extension))
        output_list.append(os.path.join(folder_name, output_extension))
       
     return data_list, output_list   
          
       
