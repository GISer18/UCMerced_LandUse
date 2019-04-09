import keras
from keras import backend as K
import h5py
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
import numpy as np 
import matplotlib.pyplot as plt
#%%
print(K.image_data_format())
model_name = '64x64-5-filter=16(bilinear)'
data_name = 'data(64x64)-train'
filepath='weights/%s.hdf5'%(model_name)

tensorboard = TensorBoard(log_dir='./logs/128x128-4-filter=32', histogram_freq=0,batch_size=128)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('weights/%s.hdf5'%(model_name), monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]#,tensorboard]#,reduce_lr]
#%%
batch_size = 16
num_classes = 4
epochs = 100    
img_size = 64
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
#%%
n_filters = 32
images = keras.layers.Input(input_shape)
shortcut1 = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(1, 1),strides=(1,1), padding="same")(images)
#conv1
seq = keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(images)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
#down1
seq = keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3),strides=(2,2), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
#conv2
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
shortcut2 = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(1, 1),strides=(1,1), padding="same")(seq)
#down2
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3),strides=(2,2), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
#conv3
seq = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
#up1, merge1
seq = keras.layers.Conv2DTranspose(filters=n_filters*4, kernel_size=(3,3), strides=(2, 2), padding='same')(seq)
seq = keras.layers.concatenate([seq,shortcut2])
seq = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
#up2, #merge2
seq = keras.layers.Conv2DTranspose(filters=n_filters*2, kernel_size=(3,3), strides=(2, 2), padding='same')(seq)
seq = keras.layers.concatenate([seq,shortcut1])
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
#intersection
intersection = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
intersection = keras.layers.BatchNormalization()(intersection)
intersection = keras.layers.Activation("relu")(intersection)
#branch1
branch1 = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding="same")(intersection)
branch1 = keras.layers.BatchNormalization()(branch1)
branch1 = keras.layers.Activation("relu")(branch1)
branch1 = keras.layers.AveragePooling2D((img_size,img_size))(branch1)
branch1 = keras.layers.Flatten()(branch1)
branch1 = keras.layers.Dense(units=num_classes,activation="sigmoid",name="class_output")(branch1)
#branch2
branch2 = keras.layers.Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(1, 1), padding='same')(intersection)
branch2 = keras.layers.BatchNormalization()(branch2)
branch2 = keras.layers.Activation("sigmoid",name="image_output")(branch2)
#model
model = keras.Model(inputs=images,outputs=[branch1,branch2])
model.summary()
#%%    
losses = {
	"image_output": "mse",
	"class_output": "binary_crossentropy",
    }
lossWeights = {
        "image_output": 1.0, 
        "class_output": 1.0
        }
optimizer = keras.optimizers.Adadelta()
model.compile(optimizer= optimizer,
              loss=losses, loss_weights=lossWeights, 
              metrics=['accuracy']
             )
#%%
x_train = x_train/255.0
x_test = x_test/255.0
print(x_train.min(),x_train.max())
print(x_test.min(),x_test.max())
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))
#%%
history = model.fit(x_train, 	
          {"image_output":x_train,"class_output":y_train},
          batch_size=8,
          epochs=epochs,
          validation_data = (x_test, {"image_output":x_test,"class_output":y_test}),
#          callbacks=callbacks_list,
          verbose=1)
#%%
#from keras.models import load_model
#model = load_model('weights/128x128-5-filter=16(bilinear).hdf5')