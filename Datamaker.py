from osgeo import gdal, osr, ogr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
import cv2
#%%
def generate_dataset_rough(data_dir,label_dir):
    #For label
    df = pd.read_excel(label_dir)
    label_df = df[class_names].copy()
    y = label_df.values
    y = y.astype(np.int8)
    #For dataset
    idx = 0
    data_sub_dir = [data_dir + '\\' +item for item in os.listdir(data_dir)]
    x = np.zeros((y.shape[0],256,256,3),dtype=np.float32)
    for folder in data_sub_dir:
        print(folder)
        image_dir = [folder + '\\' +item for item in os.listdir(folder)]
        for image_name in image_dir:
            image = gdal.Open(image_name)
            image = image.ReadAsArray()
            image = np.moveaxis(image,0,-1)
            image = cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC)
#            image = image/255.0
            x[idx] = image
            idx+=1
    return x,y
#%%
data_dir = r"D:\Thesis\Work2\UCMerced_LandUse\Images"
label_dir = r"D:\Thesis\Work2\DLRSD\multi-labels.xlsx"
ground_truth_dir = r"D:\Thesis\Work2\DLRSD\Images"
class_names = ['airplane', 'baresoil', 'buildings', 'cars', 'chaparral',
               'court', 'dock', 'field', 'grass', 'mobilehome', 'pavement', 'sand',
               'sea', 'ship', 'tanks', 'trees', 'water'] 
class_rgb_value = np.array([[166,202,240],
                            [128,128,0],
                            [0,0,128],
                            [255,0,0],
                            [0,128,0],
                            [128,0,0],
                            [255,233,233],
                            [160,160,164],
                            [0,128,128],
                            [90,87,255],
                            [255,255,0],
                            [255,192,0],
                            [0,0,255],
                            [255,0,192],
                            [128,0,128],
                            [0,255,0],
                            [0,255,255]
                            ])
#%%
x,y = generate_dataset_rough(data_dir,label_dir)
#%%
hf = h5py.File('dataset/data_notnormalize.h5', 'w')
hf.create_dataset('img',data=x)
hf.create_dataset('label',data=y)
hf.close()
#%%
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.5 , shuffle=True) 
#%%
hf = h5py.File('dataset/traintest.h5', 'w')
hf.create_dataset('x_train',data=x_train)
hf.create_dataset('x_test',data=x_test)
hf.create_dataset('y_train',data=y_train)
hf.create_dataset('y_test',data=y_test)
hf.close()










