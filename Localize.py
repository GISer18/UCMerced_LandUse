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
import cv2
import pickle
import time
import itertools
from skimage.io import imread, imsave
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score

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
    Precision =  np.zeros((1,y.shape[1]))  
    for idx in range(confusion_matrix.shape[0]):
        Precision[:,idx] = (confusion_matrix[idx,1,1])/ (confusion_matrix[idx,0,1]+confusion_matrix[idx,1,1]) 
        
    Kappa = np.zeros((1,y.shape[1]))
    for idx in range(confusion_matrix.shape[0]):
        p_0 = ((confusion_matrix[idx,0,0] + confusion_matrix[idx,0,1])/confusion_matrix[idx].sum()) * ((confusion_matrix[idx,0,0] + confusion_matrix[idx,1,0])/confusion_matrix[idx].sum())
        p_1 = ((confusion_matrix[idx,1,0] + confusion_matrix[idx,1,1])/confusion_matrix[idx].sum()) * ((confusion_matrix[idx,1,1] + confusion_matrix[idx,0,1])/confusion_matrix[idx].sum())
        p_e = p_0 + p_1
        Kappa[:,idx] = (OA[:,idx]-p_e)/(1-p_e) 
    return confusion_matrix , OA , DP , FA, Precision, Kappa
  
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
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
class_rgb_value = [[128,128,0],[255,0,0],[0,255,0],[0,255,255]]
#%%
data_name ='data(64x64)-train'
model_name = '64x64-6-filter=32'
#%%
hf = h5py.File('dataset/%s.h5'%(data_name), 'r')
print(list(hf.keys()))
hf.close()
# download data from test set
#with h5py.File('dataset/%s.h5'%(data_name), 'r') as f:
#    x_test  = f['img'][()]
#    y_test  = f['label'][()]
#    groundtruth_test = f['groundtruth'][()]
with h5py.File('dataset/%s.h5'%(data_name), 'r') as f:
    x_test  = f['x_test'][()]
    y_test  = f['y_test'][()]
    groundtruth_test = f['groundtruth_test'][()]
x_test = (x_test/255.0).astype(np.float32)
#%%
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    base_model = load_model('weights/%s.hdf5'%(model_name))
base_model.summary()
#%%
output = base_model.get_layer('class_output').output
model = keras.Model(inputs=base_model.input, outputs=output)    
model.summary()
optimizer = keras.optimizers.Adadelta()
model.compile(optimizer= optimizer,
              loss="binary_crossentropy", 
              metrics=['accuracy']
             )
#%%
score = model.evaluate(x_test, y_test,batch_size=8)
print('Loss =',score[0]) 
print('Acc  =',score[1])   
#%%
img_size = 64
CNN_reso = 64
CNN_channel = 128
ratio = img_size/CNN_reso # ratio to change resolution of output of last CNN channel
dense_weight = model.layers[-1].get_weights()[0]
#%%
y_pred = model.predict(x_test,batch_size=8)
theta = 0.5
y_pred[y_pred>=theta] = 1
y_pred[y_pred<theta] = 0
y_pred = y_pred.astype(np.int16)
cnf_matrix,OA,DP,FA,Precision, Kappa = get_confision_matrix(y_test,y_pred)
print('Classification performance\n--------------------------')
print('OA:        ',OA.mean())
print('Precision: ',Precision.mean())
print('DP,Recall: ',DP.mean())
print('FA:        ',FA.mean())
F_score = (2*Precision.mean()*DP.mean())/(Precision.mean()+DP.mean())
print('F-score:   ',F_score)
print('Kappa:     ',Kappa.mean())
#%% 123,20,1000 , fail - 1234,1150
plt.close()
i = 1234
preds_class = np.where(y_pred[i]==1)
print(y_test[i])
print(y_pred[i])
print(preds_class[0])
print('Predict =',[class_names[idx] for idx in preds_class[0]])
plt.figure(1)
plt.xlabel('Input')      
plt.imshow(x_test[i])    
#%%
preds_class = np.where(y_pred[i]==1)
print(y_test[i])
print(y_pred[i])
all_CAM = np.zeros((y_pred[i].sum(),img_size,img_size))
print('Predict =',[class_names[idx] for idx in preds_class[0]])
for idx,class_idx in enumerate(preds_class[0]):
    weight_GAP = dense_weight[:,class_idx] #weight of flatten channel
    get_last_conv_output = K.function([model.layers[0].input],
                                  [model.layers[-4].output])  #must be output from last CNN channel 
    layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
    layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
    layer_output = scipy.ndimage.zoom(layer_output, (ratio, ratio, 1), order=1) # scale the output of last CNN channel by raio dim: 112 x 112 x 128
    CAM = np.dot(layer_output.reshape((img_size*img_size, CNN_channel)), weight_GAP).reshape(img_size,img_size) # get Class Activation Map 
#    threshold = CAM.max()*0.0001 #threshold CAM
#    CAM[CAM<threshold] = 0 
    plt.figure(idx+2)
    plt.xlabel(class_names[preds_class[0][idx]])
    plt.imshow(CAM,cmap='jet')
#    CAM = (CAM-CAM.min()) / (CAM.max() - CAM.min())
#    imsave('cam.png',CAM)
#    plt.savefig('CAM.png',transparent=True)
#    break
    all_CAM[idx] = CAM
plt.figure(10)
plt.xlabel('Ground Truth')      
plt.imshow(groundtruth_test[i])
map_img = np.zeros((img_size,img_size,3),dtype = np.int16)
map_img_class = np.zeros((img_size,img_size),dtype=np.int8)
map_groundtruth_class = np.zeros((img_size,img_size),dtype=np.int8)
for x in range(img_size):
    for y in range(img_size):
        map_img[x,y,:] = class_rgb_value[preds_class[0][np.where(all_CAM[:,x,y] == all_CAM[:,x,y].max())[0][0]]]
        map_img_class[x,y] = preds_class[0][np.where(all_CAM[:,x,y] == all_CAM[:,x,y].max())[0][0]]
        map_groundtruth_class[x,y] = np.where((groundtruth_test[i,x,y] == class_rgb_value).all(axis=1))[0][0]
cnf_matrix = confusion_matrix(map_groundtruth_class.reshape(-1),map_img_class.reshape(-1),labels=[0,1,2,3])
plt.figure(11)
plt.xlabel('CAM mapping')      
plt.imshow(map_img)
#%% for K-mean clustering
preds_class = np.where(y_pred[i]==1)
print(y_test[i])
print(y_pred[i])
print('Predict =',[class_names[idx] for idx in preds_class[0]])
preds_class = np.where(y_pred[i]==1)
get_last_conv_output = K.function([model.layers[0].input],
                              [model.layers[-4].output])  #must be output from last CNN channel 
layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel

x_for_cluster = layer_output.reshape(-1,layer_output.shape[2])
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_for_cluster)
kmeans_pred = kmeans.labels_
plt.figure(4)
plt.xlabel('K-Mean')      
plt.imshow(kmeans_pred.reshape(img_size,img_size))
#%% GaussianMixture
GMM = GaussianMixture(n_components = 8 , verbose = 1)
GMM.fit(x_for_cluster)
GMM_pred = GMM.predict(x_for_cluster).reshape(256,256)
plt.figure(5)
plt.xlabel('GMM')      
plt.imshow(GMM_pred)
#%%
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(kmeans_pred, GMM_pred.reshape(-1))
#%% for collect score CAM
plt.close()
cnf_matrix = np.zeros((4,4),dtype=np.int8)
cnf_matrix = np.zeros((4,4))
kappa0 = 0
kappa1 = 0
kappa2 = 0
count = 0
count2 = 0
for i in range(x_test.shape[0]):
    preds_class = np.where(y_pred[i]==1)
    if(y_pred[i].sum()==0):
        continue
    all_CAM = np.zeros((y_pred[i].sum(),img_size,img_size))
    for idx,class_idx in enumerate(preds_class[0]):
        weight_GAP = dense_weight[:,class_idx] #weight of flatten channel
        get_last_conv_output = K.function([model.layers[0].input],
                                      [model.layers[-4].output])  #must be output from last CNN channel 
        layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
        layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
        CAM = np.dot(layer_output.reshape((img_size*img_size, CNN_channel)), weight_GAP).reshape(img_size,img_size) # get Class Activation Map 
#        CAM = CAM + np.abs(CAM.min()) # shift test
        all_CAM[idx] = CAM
    map_img_class = np.zeros((img_size,img_size),dtype=np.int8)
    map_groundtruth_class = np.zeros((img_size,img_size),dtype=np.int8)
    for x in range(img_size):
        for y in range(img_size):
            map_img_class[x,y] = preds_class[0][np.where(all_CAM[:,x,y] == all_CAM[:,x,y].max())[0][0]]
            map_groundtruth_class[x,y] = np.where((groundtruth_test[i,x,y] == class_rgb_value).all(axis=1))[0][0]
    cnf_matrix+=confusion_matrix(map_groundtruth_class.reshape(-1),map_img_class.reshape(-1),labels=[0,1,2,3])
    kappa_temp = cohen_kappa_score(map_groundtruth_class.reshape(-1),map_img_class.reshape(-1),labels=[0,1,2,3]) 
    if np.isnan(kappa_temp):
        kappa0+=0
        kappa1+=1
    else:
        kappa0+=kappa_temp
        kappa1+=kappa_temp
        kappa2+=kappa_temp
        count2+=1
    count +=1
    if i%10==0:
#        print(i,'/',x_test.shape[0])
        print(kappa0/count,kappa1/count,kappa2/count2,i,'/',x_test.shape[0])
#%%
OA = 100*(cnf_matrix[0,0] + cnf_matrix[1,1] + cnf_matrix[2,2] + cnf_matrix[3,3])/cnf_matrix.sum()
DP = 0
for i in range(cnf_matrix.shape[0]):
    DP += cnf_matrix[i,i]/cnf_matrix.sum(axis=1)[i]
DP = 100*DP/4

FA = 0
for i in range(cnf_matrix.shape[0]):
    FA += (cnf_matrix.sum(axis=0)[i]-cnf_matrix[i,i]) / (np.diag(cnf_matrix).sum() + cnf_matrix.sum(axis=0)[i] -2*cnf_matrix[i,i])
FA = 100*FA/4

recall = np.diag(cnf_matrix) / np.sum(cnf_matrix, axis = 1)
precision = np.diag(cnf_matrix) / np.sum(cnf_matrix, axis = 0)
F_score = (2*precision.mean()*recall.mean())/(precision.mean()+recall.mean())
#%%
print('Localization performance\n--------------------------')
print('OA:        ',OA)
print('Precision: ',100*precision.mean())
print('DP,Recall: ',DP)
print('FA:        ',FA)
print('F-score:   ',F_score)
print('Kappa:     ',kappa0/count,kappa1/count)
#%% machine learning
#%% generate trrain
x_for_cluster = np.zeros((420,int(256*256*0.1),256))
for i in range(x_test.shape[0]):
    get_last_conv_output = K.function([model.layers[0].input],
                              [model.layers[-4].output])  #must be output from last CNN channel 
    layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
    layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
    layer_output = layer_output.reshape(-1,256)
    x_for_cluster_train_temp, temp = train_test_split(layer_output,train_size = 0.1)
    x_for_cluster[i] = x_for_cluster_train_temp
    if i%10==0:
        print(i,'/',x_test.shape[0])
#%% 
x_for_cluster = x_for_cluster.reshape(-1,256)
kmeans = KMeans(n_clusters=4, random_state=0,verbose=1)
kmeans.fit(x_for_cluster)
GMM = GaussianMixture(n_components = 4 , verbose = 1)
GMM.fit(x_for_cluster)
#%%
count = 0
k_mean_R = 0
GM_mean_R = 0
for i in range(x_test.shape[0]):
    preds_class = np.where(y_pred[i]==1)
    if(y_pred[i].sum()==0):
        continue
    for idx,class_idx in enumerate(preds_class[0]):
        weight_GAP = dense_weight[:,class_idx] #weight of flatten channel
        get_last_conv_output = K.function([model.layers[0].input],
                                      [model.layers[-4].output])  #must be output from last CNN channel 
        layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
        layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
    map_groundtruth_class = np.zeros((img_size,img_size),dtype=np.int8)
    for x in range(img_size):
        for y in range(img_size):
            map_groundtruth_class[x,y] = np.where((groundtruth_test[i,x,y] == class_rgb_value).all(axis=1))[0][0]
    x_for_cluster = layer_output.reshape(-1,layer_output.shape[2])
    map_k_mean = kmeans.predict(x_for_cluster)
    map_GM = GMM.predict(x_for_cluster)
    
    k_mean_R += adjusted_rand_score(map_k_mean,map_groundtruth_class.reshape(-1))
    GM_mean_R+= adjusted_rand_score(map_GM,    map_groundtruth_class.reshape(-1))
    count+=1
    if i%10==0:
        print(k_mean_R/count,GM_mean_R/count,i,'/',x_test.shape[0])
#%%
i=21
    get_last_conv_output = K.function([model.layers[0].input],
                              [model.layers[-4].output])  #must be output from last CNN channel 
    layer_output = get_last_conv_output([x_test[i].reshape((1,img_size,img_size,3))])[0] #get output of last CNN channel
    layer_output = np.squeeze(layer_output) # change from 1xNxNxCNNchannel to NxNxCNNchannel
    x_for_cluster = layer_output.reshape(-1,layer_output.shape[2])
    temp_k = kmeans.predict(x_for_cluster)
    temp_GM = GMM.predict(x_for_cluster)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%% with input image
def divide_image(x,gt,size):
    new_x = np.empty((int(x.shape[0]*(np.power(int(x.shape[1]/size),2))),size,size,int(x.shape[3])),dtype=np.int16)
    new_gt = np.empty((int(x.shape[0]*(np.power(int(x.shape[1]/size),2))),size,size,int(x.shape[3])),dtype=np.int16)
    new_y = np.zeros((y.shape[0]*(np.power(int(x.shape[1]/size),2)),4),dtype=np.int8)
    for i in range(x.shape[0]):
        for j in range(int(x.shape[1]/size)):
            for k in range(int(x.shape[1]/size)):
                new_x[(np.power(int(x.shape[1]/size),2))*i + int(x.shape[1]/size)*j + k]  = x[i][size*j:size*(j+1),size*k:size*(k+1)]
                new_gt[(np.power(int(x.shape[1]/size),2))*i + int(x.shape[1]/size)*j + k] = gt[i][size*j:size*(j+1),size*k:size*(k+1)]      
    return new_x,new_gt
#%%
x_test,groundtruth_test = divide_image(x_test,groundtruth_test,img_size)
















