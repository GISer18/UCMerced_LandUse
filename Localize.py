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
#%%
model_name = 'log2'
data_name = 'data'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('weights/%s.hdf5'%(model_name))
#%%
model.summary()
dense_weight = model.layers[-1].get_weights()[0]
#%%
hf = h5py.File('dataset/%s.h5'%(data_name), 'r')
print(list(hf.keys()))
hf.close()
#%% download data from test set
with h5py.File('dataset/%s.h5'%(data_name), 'r') as f:
    x_test  = f['img'][()]
    labels  = f['y_test'][()]
img = x_test.copy()
x_test = x_test.astype(np.uint16)
img = preprocess_input(img)
#%%
score = model.evaluate(img, labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
#%%
img_size = 256
CNN_reso = 6
CNN_channel = 1536
ratio = img_size/CNN_reso # ratio to change resolution of output of last CNN channel
#%%
class_names = ['airplane', 'baresoil', 'buildings', 'cars', 'chaparral',
               'court', 'dock', 'field', 'grass', 'mobilehome', 'pavement', 'sand',
               'sea', 'ship', 'tanks', 'trees', 'water'] 
#%%
correct = 0
for i in range(1050):
    arg = model.predict(np.expand_dims(img[i],axis=0))
    arg[np.where(arg<0.5)] = 0
    arg[np.where(arg>=0.5)]= 1
    arg = arg.astype(np.int8)[0]
    if (arg==labels[i]).all():
        print(i)
        print(arg)
        print(labels[i])
        correct+=1
print(correct)
#%%
plt.imshow(x_test[1000])
#%%
i = 1000
arg = model.predict(np.expand_dims(img[i],axis=0))
arg[np.where(arg<0.5)] = 0
arg[np.where(arg>=0.5)]= 1
arg = arg.astype(np.int8)[0]
predict = np.where(arg==1)
#%%
weight_GAP = dense_weight[:,15] #weight of flatten channel
get_last_conv_output = K.function([model.layers[0].input],
                              [model.layers[-4].output])  #must be output from last CNN channel 
layer_output = get_last_conv_output([img[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
layer_output = scipy.ndimage.zoom(layer_output, (ratio, ratio, 1), order=1) # scale the output of last CNN channel by raio dim: 112 x 112 x 128
CAM = np.dot(layer_output.reshape((img_size*img_size, CNN_channel)), weight_GAP).reshape(img_size,img_size) # get Class Activation Map 
plt.imshow(CAM)





















