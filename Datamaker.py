from osgeo import gdal, osr, ogr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
import cv2
from sklearn.model_selection import train_test_split
import time
from skimage.io import imread, imsave
import warnings
#%%
def dataset_1():
    x,y,groundtruth = generate_dataset_rough(data_dir,label_dir,groundtruth_dir)
    hf = h5py.File('dataset/data.h5', 'w')
    hf.create_dataset('img',data=x)
    hf.create_dataset('label',data=y)
    hf.create_dataset('groundtruth',data=groundtruth)
    hf.close()
    x_train, x_test, y_train, y_test, groundtruth_train, groundtruth_test  = train_test_split(x, y,groundtruth, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val, groundtruth_train, groundtruth_val  = train_test_split(x_train, y_train, groundtruth_train, test_size=0.2, random_state=1)    
    hf = h5py.File('dataset/traintest.h5', 'w')
    hf.create_dataset('x_train',data=x_train)
    hf.create_dataset('x_test',data=x_test)
    hf.create_dataset('x_val',data=x_val)
    hf.create_dataset('y_train',data=y_train)
    hf.create_dataset('y_test',data=y_test)
    hf.create_dataset('y_val',data=y_val)
    hf.create_dataset('groundtruth_train',data=groundtruth_train)
    hf.create_dataset('groundtruth_test',data=groundtruth_test)
    hf.create_dataset('groundtruth_val',data=groundtruth_val)
    hf.close()
def generate_dataset_rough(data_dir,label_dir,groundtruth_dir):
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
            image=imread(image_name)
#            image = gdal.Open(image_name)
#            image = image.ReadAsArray()
#            image = np.moveaxis(image,0,-1)
            image = cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC)
            x[idx] = image
            idx+=1
    idx = 0
    groundtruth_sub_dir = [groundtruth_dir + '\\' +item for item in os.listdir(groundtruth_dir)]
    groundtruth = np.zeros((y.shape[0],256,256,3),dtype=np.float32)
    for folder in groundtruth_sub_dir:
        print(folder)
        groundtruth_label_dir = [folder + '\\' +item for item in os.listdir(folder)]
        for groundtruth_name in groundtruth_label_dir:
            image = cv2.imread(groundtruth_name)
            image = image[...,::-1]
            groundtruth[idx] = image
            idx+=1
    return x.astype(np.int16),y,groundtruth.astype(np.int16)

def extract_color(groundtruth,rgb_value):
    color = np.equal(groundtruth,rgb_value)
    color = np.logical_and(np.logical_and(color[:,:,0],color[:,:,1]),color[:,:,2]).astype(np.uint16) *1 
    return color
#%%
data_dir = r"D:\Thesis\Work2\UCMerced_LandUse\Images"
label_dir = r"D:\Thesis\Work2\DLRSD\multi-labels.xlsx"
groundtruth_dir = r"D:\Thesis\Work2\DLRSD\Images"
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
if __name__ == "__main__":
    x,y,groundtruth = generate_dataset_rough(data_dir,label_dir,groundtruth_dir) 
#%% create sub directory for each class
    data_type = ['input','groundtruth']
    if not os.path.isdir('data'):
        os.mkdir('data')
        for data in data_type:
            os.mkdir(os.path.join('data',data))
            for class_name in class_names:
                class_subdir = os.path.join('data',data,class_name)
                os.mkdir(class_subdir)
#%%
    warnings.simplefilter('ignore', UserWarning)
    class_count = np.zeros((1,17),dtype = np.uint16)
    for i in range(x.shape[0]):
        groundtruth_class = np.where(y[i]==1)[0]
        for idx in groundtruth_class:
            color_mask = np.expand_dims(extract_color(groundtruth[i],class_rgb_value[idx]),axis=2)
            masked = (x[i,:,:] * color_mask).astype(np.int16)
            save_input_dir  = os.path.join('data',data_type[0],class_names[idx],class_names[idx]+'%04d.png'%(class_count[:,idx]))
            save_ground_dir  = os.path.join('data',data_type[1],class_names[idx],class_names[idx]+'%04d.png'%(class_count[:,idx]))
            color_mask = np.squeeze(color_mask)
            imsave(save_input_dir,masked)
            imsave(save_ground_dir,color_mask*255)
            class_count[:,idx] +=1
        if i%100==0:
            print(i)
#            plt.figure(1)
#            plt.imshow(x[i])
#            plt.figure(2)
#            plt.imshow(masked)
#            plt.show()
#%%
    source_dir = os.path.join('data', 'input')
    flow_base = os.path.join('data', 'flow')
    count1 = 0
    count2 = 0
    x = np.zeros((class_count.sum(),256,256,3),dtype=np.int16)
    y = np.zeros((class_count.sum(),17),dtype=np.int8)
    class_names = os.listdir(source_dir)    
    for class_name in class_names:
        class_subdir = os.path.join(source_dir,class_name)
        print(class_subdir)
        for root, _, filenames in os.walk(class_subdir):
            for filename in filenames:
                target_name = os.path.join(class_subdir,filename)
                x[count1] = imread(target_name)
                y[count1,count2] = 1
                count1+=1
        count2+=1
    
    source_dir = os.path.join('data', 'groundtruth')
    flow_base = os.path.join('data', 'flow')
    count1 = 0
    groundtruth = np.zeros((class_count.sum(),256,256),dtype=np.int16)
    class_names = os.listdir(source_dir)    
    for class_name in class_names:
        class_subdir = os.path.join(source_dir,class_name)
        print(class_subdir)
        for root, _, filenames in os.walk(class_subdir):
            for filename in filenames:
                target_name = os.path.join(class_subdir,filename)
                groundtruth[count1] = imread(target_name)
                count1+=1
        count2+=1
#%%     
    hf = h5py.File('dataset/data2.h5', 'w')
    hf.create_dataset('img',data=x)
    hf.create_dataset('label',data=y)
    hf.create_dataset('groundtruth',data=groundtruth)
    hf.close()
    x_train, x_test, y_train, y_test, groundtruth_train,groundtruth_test = train_test_split(x, y,groundtruth,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    del x 
    del y
    del groundtruth
#%%
    hf = h5py.File('dataset/traintest2.h5', 'w')
    hf.create_dataset('x_train',data=x_train)
    hf.create_dataset('x_test',data=x_test)
    hf.create_dataset('y_train',data=y_train)
    hf.create_dataset('y_test',data=y_test)
    hf.create_dataset('groundtruth_train',data=groundtruth_train)
    hf.create_dataset('groundtruth_test',data=groundtruth_test)
    hf.close()





















