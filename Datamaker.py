from osgeo import gdal, osr, ogr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
import cv2
#%%
def generate_dataset(work_dir,label_dir):
    #For label
    df = pd.read_excel(label_dir)
    label_df = df[class_names].copy()
    y = label_df.values
    y = y.astype(np.int8)
    #For dataset
    idx = 0
    data_dir = [work_dir + '\\' +item for item in os.listdir(work_dir)]
    x = np.zeros((y.shape[0],256,256,3),dtype=np.float32)
    for folder in data_dir:
        print(folder)
        image_dir = [folder + '\\' +item for item in os.listdir(folder)]
        for image_name in image_dir:
            image = gdal.Open(image_name)
            image = image.ReadAsArray()
            image = np.moveaxis(image,0,-1)
            image = cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC)
            image = image/255.0
            x[idx] = image
            idx+=1
    return x,y
#%%
work_dir = r"D:\Thesis\Work2\UCMerced_LandUse\Images"
label_dir = r"D:\Thesis\Work2\DLRSD\multi-labels.xlsx"
class_names = ['airplane', 'baresoil', 'buildings', 'cars', 'chaparral',
               'court', 'dock', 'field', 'grass', 'mobilehome', 'pavement', 'sand',
               'sea', 'ship', 'tanks', 'trees', 'water'] 

#%%
x,y = generate_dataset(work_dir,label_dir)

#%%
hf = h5py.File('datasetUCM/data.h5', 'w')
hf.create_dataset('img',data=x)
hf.create_dataset('label',data=y)
hf.close()
#%%

#%%
#%% 