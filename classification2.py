import keras
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import h5py
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import pickle
#%%
print(K.image_data_format())
model_name = 'model_ratio_3'
data_name = 'traintest4'
filepath='weights/%s.hdf5'%(model_name)

#tensorboard = TensorBoard(log_dir='./logs/downclass4', histogram_freq=0,batch_size=128)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('weights/%s.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_name), monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]#tensorboard]#,reduce_lr]
#%%
batch_size = 8
num_classes = 4
epochs = 150
img_size = 256
input_shape = (img_size, img_size, 3)
#%% load test
hf = h5py.File('dataset/%s.h5'%(data_name), 'r')
print(list(hf.keys()))
hf.close()
#%%
with h5py.File('dataset/%s.h5'%(data_name), 'r') as f:
    x_train = f['x_train'][()]
    x_test = f['x_test'][()]
    y_train = f['y_train'][()]
    y_test = f['y_test'][()]
#%% simple model
images = keras.layers.Input(input_shape)
shortcut1 = keras.layers.Conv2D(filters=256, kernel_size=(1, 1),strides=(1,1), padding="same")(images)

net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(images)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3),strides=(2,2), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.Conv2D(filters=128, kernel_size=(3, 3),strides=(2,2), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)


net = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(net)

#net = keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(net)

net = keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2, 2), padding='same')(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)


net = keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2, 2), padding='same')(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)


net = keras.layers.add([net,shortcut1])
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)


net = keras.layers.AveragePooling2D((256,256))(net)
net = keras.layers.Flatten()(net)

net = keras.layers.Dense(units=num_classes,activation="linear")(net)

model = keras.Model(inputs=images,outputs=net)
model.summary()
#%%    
#optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = keras.optimizers.Adadelta()
model.compile(optimizer= optimizer,
              loss = 'mean_squared_error',         
             )
# In[10]:
x_train = x_train/255.0
x_test = x_test/255.0
print(x_train.min(),x_train.max())
print(x_test.min(),x_test.max())
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))
#%%
history = model.fit(x_train, y_train, 
          batch_size=batch_size,
          epochs=epochs,
          validation_data = (x_test, y_test),
          callbacks=callbacks_list,
          verbose=1)
with open('weights/%s.pickle'%(model_name), 'wb') as file_pi:
    pickle.dump(history.history, file_pi) 
#%%

    
    
    
    
    
    
    
    
    
    
    