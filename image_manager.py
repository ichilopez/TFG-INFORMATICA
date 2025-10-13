import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
path_meta = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/meta.csv'
path_dicom = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/dicom_info.csv'
path_calcification_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/calc_case_description_test_set.csv'
path_calcification_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/calc_case_description_train_set.csv'
path_mass_test = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_test_set.csv'
path_mass_train = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/csv/mass_case_description_train_set.csv'


def charge_df ():
    df_meta = pd.read_csv(path_meta)
    dicom_data = pd.read_csv(path_dicom)
    calcification_test_data = pd.read_csv(path_calcification_test)
    calcification_train_data = pd.read_csv(path_calcification_train)
    mass_test_data = pd.read_csv(path_mass_test)
    mass_train_data = pd.read_csv(path_mass_train)
    calcification_train_data.reset_index(drop=True, inplace=True)
    calcification_train_data.drop(1216, inplace=True)
    calcification_train_data.reset_index(drop=True, inplace=True)
    return [df_meta,dicom_data,calcification_train_data,calcification_test_data,mass_test_data,mass_train_data]


def is_roi_mask(image_path, tol=5):
    """
    Determina si una imagen es una ROI-mask.
    Una ROI-mask típica tiene casi todos los píxeles negros (0)
    y una pequeña proporción de píxeles blancos (~255).
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    total = arr.size
    negros = np.sum(arr <= tol)
    blancos = np.sum(arr >= 255 - tol)

    porcentaje_negro = (negros / total) * 100
    porcentaje_blanco = (blancos / total) * 100
    
    # ROI-mask: muchísimos negros y pocos blancos
    if porcentaje_negro > 90 and 0.01 < porcentaje_blanco < 10:
        return True
    else:
        return False


def renombrar_imagenes_carpeta(carpeta):
    """
    Recorre todos los archivos de la carpeta y renombra:
    - '1.jpeg' si es imagen mamaria
    - '2.jpeg' si es ROI-mask
    """
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)

        # Saltar si no es archivo
        if not os.path.isfile(ruta):
            continue

        if is_roi_mask(ruta):
            nuevo_nombre = "2.jpeg"  # ROI-mask
        else:
            nuevo_nombre = "1.jpeg"  # Imagen mamaria

        nueva_ruta = os.path.join(carpeta, nuevo_nombre)

        # 🔹 Evitar sobrescribir si ya existe
        if os.path.exists(nueva_ruta):
            print(f"⚠️ Ya existe {nuevo_nombre} en {carpeta}, se omite {archivo}")
            continue

        else :
            os.rename(ruta, nueva_ruta)
            print(f"✅ Renombrado: {archivo} → {nuevo_nombre}")


    def getMassPathList(mass_test_data,mass_train_data):
        root = 'C:/Users/Itziar/Documents/Documentos/TFG-INF-DATOS/archive/jpeg'
        train_paths_list = []
        test_paths_list = []
        masks_train = []
        masks_test = []
        for i in len(mass_test_data):
            aux_list = mass_test_data.loc[i,'ROI mask file path'].split('/')
            folder_name = os.path.join(root,aux_list[-2])
            train_paths_list.append(os.path.join(folder_name,'1.jpeg'))
            masks_train.append(os.path.join(folder_name,'2.jpeg'))







