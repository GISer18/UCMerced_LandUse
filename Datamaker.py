from osgeo import gdal, osr, ogr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
import cv2
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import warnings
from sklearn.utils import class_weight

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
def dataset_2(class_names):
    #separate class
    x,y,groundtruth = generate_dataset_rough(data_dir,label_dir,groundtruth_dir) 
    data_type = ['input','groundtruth']
    if not os.path.isdir('data'):
        os.mkdir('data')
        for data in data_type:
            os.mkdir(os.path.join('data',data))
            for class_name in class_names:
                class_subdir = os.path.join('data',data,class_name)
                os.mkdir(class_subdir)
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

    source_dir = os.path.join('data', 'input')
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
    hf = h5py.File('dataset/traintest2.h5', 'w')
    hf.create_dataset('x_train',data=x_train)
    hf.create_dataset('x_test',data=x_test)
    hf.create_dataset('y_train',data=y_train)
    hf.create_dataset('y_test',data=y_test)
    hf.create_dataset('groundtruth_train',data=groundtruth_train)
    hf.create_dataset('groundtruth_test',data=groundtruth_test)
    hf.close()
    
def dataset_3(class_names):
    #combine and separate class
    x,y,groundtruth = generate_dataset_rough(data_dir,label_dir,groundtruth_dir) 
    print(class_names[2],class_names[10],class_names[-2],'X')
    new_y = np.zeros((y.shape[0],4),dtype = np.int8)
    new_class_names = [class_names[2],class_names[10],class_names[-2],'X'] 
    new_class_rgb_value = np.vstack((class_rgb_value[2],class_rgb_value[10],class_rgb_value[-2]))
    for idx,item in enumerate(y):
        count = 0
        for value in item:
            count+=value
        count-=sum([item[2],item[10],item[-2]])
        new_y[idx,0] = item[2]
        new_y[idx,1] = item[10]
        new_y[idx,2] = item[-2]
        if(count>0):
            new_y[idx,3] = 1
    hf = h5py.File('dataset/full_data3.h5', 'w')
    hf.create_dataset('img',data=x)
    hf.create_dataset('label',data=new_y)
    hf.create_dataset('groundtruth',data=groundtruth)
    hf.close()
    data_type = ['partial_input','partial_groundtruth']
    if not os.path.isdir('data3'):
        os.mkdir('data3')
        for data in data_type:
            os.mkdir(os.path.join('data3',data))
            for class_name in new_class_names:
                class_subdir = os.path.join('data3',data,class_name)
                os.mkdir(class_subdir)
    warnings.simplefilter('ignore', UserWarning)
    class_count = np.zeros((1,4),dtype = np.uint16)
    for i in range(new_y.shape[0]):
        groundtruth_class = np.where(new_y[i]==1)[0]
        for idx in groundtruth_class:
            if idx!=3:
                color_mask = np.expand_dims(extract_color(groundtruth[i],new_class_rgb_value[idx]),axis=2)
                masked = (x[i,:,:] * color_mask).astype(np.int16)
            else:
                masked = np.zeros_like(x[i,:,:])
                other_location = np.where(y[i]==1)[0]
                for location in other_location:
                    if (location!=2 and location!=10 and location!=15):
                        color_mask = np.expand_dims(extract_color(groundtruth[i],class_rgb_value[location]),axis=2)
                        temp_masked = (x[i,:,:] * color_mask).astype(np.int16)   
                        masked+=temp_masked     
            save_input_dir  = os.path.join('data3',data_type[0],new_class_names[idx],new_class_names[idx]+'%04d.png'%(class_count[:,idx]))
            save_ground_dir  = os.path.join('data3',data_type[1],new_class_names[idx],new_class_names[idx]+'%04d.png'%(class_count[:,idx]))
            color_mask = np.squeeze(color_mask)
            imsave(save_input_dir,masked)
            imsave(save_ground_dir,color_mask*255)
            class_count[:,idx] +=1
        if i%100==0:
            print(i)

    source_dir = os.path.join('data3', 'partial_input')
    count1 = 0
    count2 = 0
    x = np.zeros((class_count.sum(),256,256,3),dtype=np.int16)
    y = np.zeros((class_count.sum(),4),dtype=np.int8)
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
    
    source_dir = os.path.join('data3', 'partial_groundtruth')
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
    hf = h5py.File('dataset/partial_data3.h5', 'w')
    hf.create_dataset('img',data=x)
    hf.create_dataset('label',data=y)
    hf.create_dataset('groundtruth',data=groundtruth)
    hf.close()
    with h5py.File(r"D:\Thesis\Work2\dataset\full_data3.h5", 'r') as f:
        x = f['img'][()]
        y = f['label'][()]
        groundtruth = f['groundtruth'][()]
    x_train, x_test, y_train, y_test, groundtruth_train,groundtruth_test = train_test_split(x, y,groundtruth,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    hf = h5py.File('dataset/full_traintest3.h5', 'w')
    hf.create_dataset('x_train',data=x_train)
    hf.create_dataset('x_test',data=x_test)
    hf.create_dataset('y_train',data=y_train)
    hf.create_dataset('y_test',data=y_test)
    hf.create_dataset('groundtruth_train',data=groundtruth_train)
    hf.create_dataset('groundtruth_test',data=groundtruth_test)
    hf.close()
    with h5py.File(r"D:\Thesis\Work2\dataset\partial_data3.h5", 'r') as f:
        x = f['img'][()]
        y = f['label'][()]
        groundtruth = f['groundtruth'][()]
    x_train, x_test, y_train, y_test, groundtruth_train,groundtruth_test = train_test_split(x, y,groundtruth,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    hf = h5py.File('dataset/partial_traintest3.h5', 'w')
    hf.create_dataset('x_train',data=x_train)
    hf.create_dataset('x_test',data=x_test)
    hf.create_dataset('y_train',data=y_train)
    hf.create_dataset('y_test',data=y_test)
    hf.create_dataset('groundtruth_train',data=groundtruth_train)
    hf.create_dataset('groundtruth_test',data=groundtruth_test)
    hf.close()

        
def dateset_4(class_names): # just combine classes
    #combine class [0,2,3,5,6,9,10,13,14] into 'man_made'
    #combine class [1,11] into 'land'
    #combine class [4,7,8,15] into 'plant'
    #combine class [12,16] into 'water'
    x,y,groundtruth = generate_dataset_rough(data_dir,label_dir,groundtruth_dir) 
    land       = [1,11]
    man_made   = [0,2,3,5,6,9,10,13,14]
    vegetation = [4,7,8,15]
    water      = [12,16]
    classes = [land,man_made,vegetation,water]
    print([class_names[item] for item in land])
    print([class_names[item] for item in man_made])
    print([class_names[item] for item in vegetation])
    print([class_names[item] for item in water])
    class_names = ['land','man_made','vegetation','water']
    combine_class_rgb_value = [class_rgb_value[item] for item in classes]
    change_rgb = [[128,128,0],[255,0,0],[0,255,0],[0,255,255]]

    new_y = np.zeros((y.shape[0],4),dtype=np.int8)
    for i in range(y.shape[0]):
        for idx,item in enumerate(classes):
            if ((y[i,item]==1).any()):
                new_y[i,idx] = 1   
    combine_groundtruth = np.zeros_like(groundtruth)
    for i in range(groundtruth.shape[0]):
        # find what color in groundtruth, then map these color to target color
        color_in_groundtruth = np.unique(groundtruth[i].reshape(-1,groundtruth[i].shape[2]),axis=0)
        for item in color_in_groundtruth:
            #find which target class
            for idx,rgb_set in enumerate(combine_class_rgb_value):
                for rgb in rgb_set:
                    if((rgb == item).all()):
                        #change the color
                        temp = extract_color(groundtruth[i],item)
                        temp = temp.reshape(temp.shape[0],temp.shape[1],1) * change_rgb[idx]
                        combine_groundtruth[i] += temp
        if i%100==0:
            print(i)
    hf = h5py.File('dataset/data4.h5', 'w')
    hf.create_dataset('img',data=x)
    hf.create_dataset('label',data=new_y)
    hf.create_dataset('groundtruth',data=combine_groundtruth)
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

def generate_groundtrurh_ratio(groundtruth,class_rgb_value,size=256):
    groundtruth_resample = [cv2.resize(groundtruth[i] , (size,size),interpolation = cv2.INTER_NEAREST) for i in range(groundtruth.shape[0])]
    groundtruth_resample = np.stack(groundtruth_resample, axis=0)
    class_ratio = np.zeros_like(y,dtype=np.float32)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            color = extract_color(groundtruth_resample[i],class_rgb_value[j])
            class_ratio[i,j] = np.count_nonzero(color)
        print(i,'/',y.shape[0])
    class_ratio = class_ratio/(groundtruth_resample.shape[1] * groundtruth_resample.shape[2])
    return class_ratio

def divide_image(x,y,gt,change_rgb,size):
    new_x = np.empty((int(x.shape[0]*(np.power(int(x.shape[1]/size),2))),size,size,int(x.shape[3])),dtype=np.int16)
    new_gt = np.empty((int(x.shape[0]*(np.power(int(x.shape[1]/size),2))),size,size,int(x.shape[3])),dtype=np.int16)
    new_y = np.zeros((y.shape[0]*(np.power(int(x.shape[1]/size),2)),4),dtype=np.int8)
    for i in range(x.shape[0]):
        for j in range(int(x.shape[1]/size)):
            for k in range(int(x.shape[1]/size)):
                new_x[(np.power(int(x.shape[1]/size),2))*i + int(x.shape[1]/size)*j + k]  = x[i][size*j:size*(j+1),size*k:size*(k+1)]
                new_gt[(np.power(int(x.shape[1]/size),2))*i + int(x.shape[1]/size)*j + k] = gt[i][size*j:size*(j+1),size*k:size*(k+1)]      
                for l in range(len(change_rgb)):
                    if (extract_color(new_gt[(np.power(int(x.shape[1]/size),2))*i + int(x.shape[1]/size)*j + k],change_rgb[l]).sum()) != 0:
                        new_y[(np.power(int(x.shape[1]/size),2))*i + int(x.shape[1]/size)*j + k][l] = 1
    return new_x,new_y,new_gt
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
                            ],dtype=np.int16)
#%% 
if __name__ == "__main__":
    data_name = 'data4'
    with h5py.File('dataset/%s.h5'%(data_name), 'r') as f:
        x = f['img'][()]
        y = f['label'][()]
        gt = f['groundtruth'][()]
    land       = [1,11]
    man_made   = [0,2,3,5,6,9,10,13,14]
    vegetation = [4,7,8,15]
    water      = [12,16]
    classes = [land,man_made,vegetation,water]
    print([class_names[item] for item in land])
    print([class_names[item] for item in man_made])
    print([class_names[item] for item in vegetation])
    print([class_names[item] for item in water])
    class_names = ['land','man_made','vegetation','water']
    combine_class_rgb_value = [class_rgb_value[item] for item in classes]
    change_rgb = [[128,128,0],[255,0,0],[0,255,0],[0,255,255]]
#%%
    size=64
    new_x,new_y,new_gt = divide_image(x,y,gt,change_rgb,size)
#%%
    for i in range(0,new_x.shape[0]):
        plt.figure(figsize=(5,5))
        plt.subplot(2,3,1)
        plt.imshow(new_x[i])
        plt.subplot(2,3,3)
        plt.imshow(new_gt[i])
        plt.subplot(2,3,5)
        plt.imshow(x[int(i/(np.power(int(x.shape[1]/size),2)))])
        print(new_y[i])
        plt.pause(1)
        plt.close()
#%%
    data_name = 'data(64x64)'
    hf = h5py.File('dataset/%s.h5'%(data_name), 'w')
    hf.create_dataset('img',data=new_x)
    hf.create_dataset('label',data=new_y)
    hf.create_dataset('groundtruth',data=new_gt)
    hf.close()
#%%
    with h5py.File(r"D:\Thesis\Work2\dataset\data(64x64).h5", 'r') as f:
        x = f['img'][()]
        y = f['label'][()]
        groundtruth = f['groundtruth'][()]

    x_train, x_test, y_train, y_test, groundtruth_train,groundtruth_test = train_test_split(x, y,groundtruth,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    del x 
    del y
    del groundtruth
    hf = h5py.File('dataset/data(64x64)-train.h5', 'w')
    hf.create_dataset('x_train',data=x_train)
    hf.create_dataset('x_test',data=x_test)
    hf.create_dataset('y_train',data=y_train)
    hf.create_dataset('y_test',data=y_test)
    hf.create_dataset('groundtruth_train',data=groundtruth_train)
    hf.create_dataset('groundtruth_test',data=groundtruth_test)
    hf.close()
    
    
    
    
    
    
    
    
    
    
    
    