from osgeo import gdal, osr, ogr
import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model
from keras import backend as K
import h5py
import matplotlib.pyplot as plt
import numpy as np
import maxflow
import scipy
from keras.applications.inception_resnet_v2 import preprocess_input
import cv2
import pickle
import time

def get_confision_matrix(y,preds):
    confusion_matrix = np.zeros((preds.shape[1],2,2))
    for pred_idx in range(len(y)):
        for class__idx in range((y.shape[1])):
            if y[pred_idx,class__idx] == 0:
                if y_test[pred_idx,class__idx] == preds[pred_idx,class__idx]:
                    confusion_matrix[class__idx,0,0] += 1
                else:
                    confusion_matrix[class__idx,0,1] += 1
            else:
                if y[pred_idx,class__idx] == preds[pred_idx,class__idx]:
                    confusion_matrix[class__idx,1,1] += 1
                else:
                    confusion_matrix[class__idx,1,0] += 1
    OA = np.zeros((1,y.shape[1]))
    for idx in range(confusion_matrix.shape[0]):
        OA[:,idx] = (confusion_matrix[idx,0,0] + confusion_matrix[idx,1,1])/ (confusion_matrix[idx].sum())       
    DP = np.zeros((1,y.shape[1]))
    for idx in range(confusion_matrix.shape[0]):
        DP[:,idx] = (confusion_matrix[idx,1,1])/ (confusion_matrix[idx,1,0]+confusion_matrix[idx,1,1]) 
    FA = np.zeros((1,y.shape[1]))
    for idx in range(confusion_matrix.shape[0]):
        FA[:,idx] = (confusion_matrix[idx,0,1])/ (confusion_matrix[idx,0,0]+confusion_matrix[idx,0,1])
    Kappa = np.zeros((1,y.shape[1]))
    for idx in range(confusion_matrix.shape[0]):
        p_0 = ((confusion_matrix[idx,0,0] + confusion_matrix[idx,0,1])/confusion_matrix[idx].sum()) * ((confusion_matrix[idx,0,0] + confusion_matrix[idx,1,0])/confusion_matrix[idx].sum())
        p_1 = ((confusion_matrix[idx,1,0] + confusion_matrix[idx,1,1])/confusion_matrix[idx].sum()) * ((confusion_matrix[idx,1,1] + confusion_matrix[idx,0,1])/confusion_matrix[idx].sum())
        p_e = p_0 + p_1
        Kappa[:,idx] = (OA[:,idx]-p_e)/(1-p_e) 
    return confusion_matrix , OA , DP , FA, Kappa
  
def extract_color(groundtruth,rgb_value):
    color = np.equal(groundtruth,rgb_value)
    color = np.logical_and(np.logical_and(color[:,:,0],color[:,:,1]),color[:,:,2]).astype(np.uint16) *255 
    return color
  
def generate_groundtruth_class_pixel(groundtruth,size):
    groundtruth_resample = [cv2.resize(groundtruth[i] , (size,size),interpolation = cv2.INTER_NEAREST) for i in range(groundtruth.shape[0])]
    groundtruth_resample = np.stack(groundtruth_resample, axis=0)
    groundtruth_pixel_class = np.zeros(( groundtruth_resample.shape[0] * groundtruth_resample.shape[1] * groundtruth_resample.shape[2]  ,17),dtype = np.uint8)
    for i in range(groundtruth_resample.shape[0]):
        for j in range(groundtruth_resample.shape[1]):
            for k in range(groundtruth_resample.shape[2]):
                cond = groundtruth_resample[i,j,k] == class_rgb_value
                cond = np.logical_and(np.logical_and(cond[:,0] , cond[:,1]) , cond[:,2]).astype(np.int8)
                groundtruth_pixel_class[ (groundtruth_resample.shape[2]*groundtruth_resample.shape[1])*i + groundtruth_resample.shape[2]*j + k] = cond
        print(i,'/',groundtruth_resample.shape[0])
    return groundtruth_pixel_class
  
def generate_heatmap_pixel(x,model):
  heatmap_pixel = np.zeros(( x.shape[0] * CNN_reso * CNN_reso ,CNN_channel),dtype = np.float32)
  last_conv_output = K.function([model.layers[0].input],
                                [model.layers[-4].output])
  for i in range(x.shape[0]):
    layer_output = last_conv_output( [np.expand_dims(x[i],axis=0) ])[0]
    layer_output = np.squeeze(layer_output)
    for j in range(layer_output.shape[0]):
      for k in range(layer_output.shape[1]):
        heatmap_pixel[(layer_output.shape[0] * layer_output.shape[1] * i) +  layer_output.shape[1] * j + k] = layer_output[j,k]
    print(i,'/',x.shape[0])
  return heatmap_pixel

def generate_groundtrurh_ratio(groundtruth,size):
    groundtruth_resample = [cv2.resize(groundtruth[i] , (size,size),interpolation = cv2.INTER_NEAREST) for i in range(groundtruth.shape[0])]
    groundtruth_resample = np.stack(groundtruth_resample, axis=0)
    class_ratio = np.zeros_like(y_test,dtype=np.float32)
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            color = extract_color(groundtruth_resample[i],class_rgb_value[j])
            class_ratio[i,j] = np.count_nonzero(color)
        print(i,'/',y_test.shape[0])
    class_ratio = class_ratio/(groundtruth_resample.shape[1] * groundtruth_resample.shape[2])
    return class_ratio
  
def generate_heatmap(x,model):
  heatmap = np.zeros((x.shape[0],CNN_reso,CNN_reso,CNN_channel),dtype = np.float32)
  last_conv_output = K.function([model.layers[0].input],
                                  [model.layers[-4].output])
  for i in range (x.shape[0]):
    layer_output = last_conv_output( [np.expand_dims(x[i],axis=0) ])[0]
    layer_output = np.squeeze(layer_output)
    heatmap[i] = layer_output
    print(i,'/',x.shape[0])
  return heatmap
#%%
channel_means = np.load('mean.npy')
channel_stds = np.load('std.npy')
model_name = 'log14.142-0.24'
data_name = 'traintest'
hf = h5py.File('dataset/%s.h5'%(data_name), 'r')
print(list(hf.keys()))
hf.close()
#%% download data from test set
with h5py.File('dataset/%s.h5'%(data_name), 'r') as f:
    x_test  = f['x_val'][()]
    y_test  = f['y_val'][()]
    groundtruth_test = f['groundtruth_val'][()]
x_test = x_test.astype(np.float32)
x_test = (x_test-channel_means)/channel_stds
x_test = (x_test-x_test.min())/(x_test.max()-x_test.min())
x_test = x_test.astype(np.float32)
#%%
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('weights/%s.hdf5'%(model_name) , custom_objects={"weighted_loss": 'binary_crossentropy'})
#with open('weights/%s.pickle'%(model_name), 'rb') as input_file:
#    e = pickle.load(input_file)
model.summary()
#%%
img_size = 256
CNN_reso = 64
CNN_channel = 256
ratio = img_size/CNN_reso # ratio to change resolution of output of last CNN channel

dense_weight = model.layers[-1].get_weights()[0]

preds = model.predict(x_test,batch_size=16)

preds[preds>=0.5] = 1
preds[preds<0.5] = 0
preds = preds.astype(np.int16)

confusion_matrix,OA,DP,FA, Kappa = get_confision_matrix(y_test,preds)
print('Classification performance\n--------------------------')
print('OA:    ',OA.mean())
print('DP:    ',DP.mean())
print('FA:    ',FA.mean())
print('Kappa: ',Kappa.mean())

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
i = 3
preds_class = np.where(preds[i]==1)
print(y_test[i])
print(preds[i])
print(preds_class[0])
print('Predict =',[class_names[idx] for idx in preds_class[0]])
plt.imshow(x_test[i])
#%%
confusion_matrix_localize = np.zeros((y_test.shape[1],2,2))
#plt.figure(1)
#plt.imshow(x_test[i])
for i in range(len(y_test)):
    preds_class = np.where(preds[i]==1)
    print(i,'/',len(y_test))
    for idx,class_idx in enumerate(preds_class[0]):
    #    plt.figure(idx+2)
    #    plt.xlabel(class_names[class_idx])
        weight_GAP = dense_weight[:,class_idx] #weight of flatten channel
        get_last_conv_output = K.function([model.layers[0].input],
                                      [model.layers[-4].output])  #must be output from last CNN channel 
        layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
        layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
        layer_output = scipy.ndimage.zoom(layer_output, (ratio, ratio, 1), order=1) # scale the output of last CNN channel by raio dim: 112 x 112 x 128
        CAM = np.dot(layer_output.reshape((img_size*img_size, CNN_channel)), weight_GAP).reshape(img_size,img_size) # get Class Activation Map 
    #    plt.imshow(CAM,cmap='jet')
       #background/foreground segment by graphcut
        CAM = 255 * (CAM-CAM.min())/(CAM.max() - CAM.min()) #normalize CAM to [0,255]
        g = maxflow.Graph[int]()
        nodeids = g.add_grid_nodes(CAM.shape)
        g.add_grid_edges(nodeids, 1) 
        g.add_grid_tedges(nodeids, CAM, 255-CAM) 
        g.maxflow()
        sgm = g.get_grid_segments(nodeids)
        sgm = np.int_(np.logical_not(sgm))
    #    plt.imshow(sgm,cmap='gray',alpha=1) #white is foreground
        location = np.where(sgm==1)
        mask = np.zeros((sgm.shape[0],sgm.shape[1],3),dtype=np.int16)
        mask[location[0],location[1],:] = class_rgb_value[class_idx]
    #    plt.imshow(mask)   
        for x in range(256):
            for y in range(256):
                if(  (groundtruth_test[i][x,y,:] == class_rgb_value[class_idx]).all()  ):
                    if(  (mask[x,y,:] == class_rgb_value[class_idx]).all()  ):
                        confusion_matrix_localize[class_idx,1,1] += 1
                    else:
                        confusion_matrix_localize[class_idx,1,0] += 1
                else:
                    if(  (mask[x,y,:] == class_rgb_value[class_idx]).all()  ):
                        confusion_matrix_localize[class_idx,0,1] += 1
                    else:
                        confusion_matrix_localize[class_idx,1,0] += 1
#plt.figure(idx+3)
#plt.xlabel('GroundTruth')
#plt.imshow(groundtruth_test[i])
np.save('log11.npy',confusion_matrix_localize)        
#%%
print(confusion_matrix_localize)
#%%
i = 0
preds_class = np.where(preds[i]==1)
print(preds[i])
print(y_test[i])
print('Predict =',[class_names[idx] for idx in preds_class[0]])
plt.imshow(x_test[i])
preds_class = np.where(preds[i]==1)
for idx,class_idx in enumerate(preds_class[0]):
    plt.figure(idx+2)
    plt.xlabel(class_names[class_idx])
    weight_GAP = dense_weight[:,class_idx] #weight of flatten channel
    get_last_conv_output = K.function([model.layers[0].input],
                                  [model.layers[-4].output])  #must be output from last CNN channel 
    layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
    layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
    layer_output = scipy.ndimage.zoom(layer_output, (ratio, ratio, 1), order=1) # scale the output of last CNN channel by raio dim: 112 x 112 x 128
    CAM = np.dot(layer_output.reshape((img_size*img_size, CNN_channel)), weight_GAP).reshape(img_size,img_size) # get Class Activation Map 
    plt.imshow(CAM,cmap='jet')
   #background/foreground segment by graphcut
    CAM = 255 * (CAM-CAM.min())/(CAM.max() - CAM.min()) #normalize CAM to [0,255]
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes(CAM.shape)
    g.add_grid_edges(nodeids, 1) 
    g.add_grid_tedges(nodeids, CAM, 255-CAM) 
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    sgm = np.int_(np.logical_not(sgm))
#    plt.imshow(sgm,cmap='gray',alpha=1) #white is foreground
    location = np.where(sgm==1)
    mask = np.zeros((sgm.shape[0],sgm.shape[1],3),dtype=np.int16)
    mask[location[0],location[1],:] = class_rgb_value[class_idx]      
    plt.imshow(mask)
plt.figure(idx+3)
plt.imshow(groundtruth_test[i])
#%% generate_heatmap_pixel
heatmap = generate_heatmap(x_test,model)
groundtruth_ratio = generate_groundtrurh_ratio(groundtruth_test,CNN_reso)
np.save('heatmap.npy',heatmap)
np.save('groundtruth_class_ratio.npy',groundtruth_ratio)















