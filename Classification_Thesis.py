import keras
from keras import backend as K
import h5py
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
import numpy as np 
import matplotlib.pyplot as plt
#%%
print(K.image_data_format())
model_name = '128x128-8'
data_name ='data(128x128)-train'
filepath='weights/%s.hdf5'%(model_name)
#tensorboard = TensorBoard(log_dir='./logs/64x64-7-filter=72(branchwithaugment)', histogram_freq=0,batch_size=128)  
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint_class_loss = ModelCheckpoint('weights/%s_class_loss.hdf5'%(model_name), monitor='val_class_output_loss', verbose=0, save_best_only=True, mode='min')
#checkpoint_image_loss = ModelCheckpoint('weights/%s_image_loss.hdf5'%(model_name), monitor='val_image_output_loss', verbose=0, save_best_only=True, mode='min')
checkpoint_class_acc = ModelCheckpoint('weights/%s_class_acc.hdf5'%(model_name), monitor='val_class_output_acc', verbose=0, save_best_only=True, mode='max')
#checkpoint_image_acc = ModelCheckpoint('weights/%s_image_acc.hdf5'%(model_name), monitor='val_image_output_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint_class_loss,checkpoint_class_acc]#checkpoint_image_loss,checkpoint_class_acc,checkpoint_image_acc]#,tensorboard]#,reduce_lr]
#%%
batch_size = 16
num_classes = 4
epochs = 150    
img_size = 128
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
n_filters = 64
images = keras.layers.Input(input_shape)
shortcut1 = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(1, 1),strides=(1,1), padding="same")(images)
shortcut1 = keras.layers.BatchNormalization()(shortcut1)
shortcut1 = keras.layers.Activation("relu")(shortcut1)
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
shortcut2 = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(1, 1),strides=(1,1), padding="same")(seq)
shortcut2 = keras.layers.BatchNormalization()(shortcut2)
shortcut2 = keras.layers.Activation("relu")(shortcut2)
#conv2
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
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
seq = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.add([seq,shortcut2])
#convT1
seq = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)  
#up2, #merge2
seq = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.add([seq,shortcut1])
#convT2
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
seq = keras.layers.Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding="same")(seq)
seq = keras.layers.BatchNormalization()(seq)
seq = keras.layers.Activation("relu")(seq)
#intersection
intersection = keras.layers.Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding="same")(seq)
intersection = keras.layers.BatchNormalization()(intersection)
intersection = keras.layers.Activation("relu")(intersection)
#branch1
branch1 = keras.layers.AveragePooling2D((img_size,img_size))(intersection)
branch1 = keras.layers.Flatten()(branch1)
branch1 = keras.layers.Dense(units=num_classes,activation="sigmoid",name="class_output")(branch1)
#branch2
branch2 = keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1, 1), padding='same')(intersection)
branch2 = keras.layers.BatchNormalization()(branch2)    
branch2 = keras.layers.Activation("relu",name="image_output")(branch2)
#model
model = keras.Model(inputs=images,outputs=[branch1,branch2])
model.summary()
#%%    
losses = {
	"image_output": "mse",
	"class_output": "binary_crossentropy",
    }
lossWeights = {
        "image_output": 1, 
        "class_output": 1
        }
optimizer = keras.optimizers.Adadelta()
model.compile(optimizer= optimizer,
              loss=losses, loss_weights=lossWeights, 
              metrics=['accuracy']
             )
#%%
x_train = x_train.astype(np.uint8)
x_test  = x_test.astype(np.uint8)
print(x_train.min(),x_train.max())
print(x_test.min(),x_test.max())
#%%
#import imgaug.augmenters as iaa
#batch_size_aug = 5096
#aug_mal = 2 # 2 pow of aug_mul
#seq = iaa.Sequential([
#  iaa.Fliplr(0.5),
#  iaa.Flipud(0.5),
#  iaa.Rot90((0,5)),
##  iaa.LogContrast((0.9,1.1), 0.5),
##  iaa.GammaContrast((0.9,1.1), 0.5),
##  iaa.PiecewiseAffine(scale=(0.01, 0.015)),
#])
#seq.show_grid([x_train[0], x_train[1], x_train[2], x_train[3]], cols=8, rows=8)
##%%
#def Image_Augmentation(x, seq, batch_size):
#    for cbatch in range (0, x.shape[0], batch_size):
#        yield seq.augment_images(x[cbatch:cbatch+batch_size])
##%%
#for round in range(aug_mal):
#    images_aug = Image_Augmentation(x_train,seq,batch_size_aug)
#    x_train_aug = np.zeros_like(x_train)
#    for idx,image_batch in enumerate(images_aug):
#        print(idx, image_batch.shape)
#        x_train_aug[idx*batch_size_aug:(idx+1) * batch_size_aug] = image_batch
#    x_train = np.concatenate((x_train,x_train_aug), axis=0)
#    y_train = np.concatenate((y_train,y_train),axis=0)
#    print(x_train.shape, y_train.shape)
#del x_train_aug
#%%
x_train = (x_train/255.0).astype(np.float32)
x_test = (x_test/255.0).astype(np.float32)
print(x_train.min(),x_train.max())
print(x_test.min(),x_test.max())
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))
#%%
class_weight = (x_train.shape[0]) / (num_classes * y_train.sum(axis=0))
class_weight = class_weight.tolist()
#%%
history = model.fit(x_train, 	
          {"image_output":x_train,"class_output":y_train},
          batch_size = batch_size,
          epochs = epochs,
          validation_data = (x_test, {"image_output":x_test,"class_output":y_test}),
          callbacks = callbacks_list,
          class_weight = {"class_output":class_weight},
          verbose=1)
#%%
#from keras.models import load_model
#model = load_model('weights/128x128-5-filter=16(bilinear).hdf5')
history.history.keys()
plt.plot(history.history['val_class_output_acc'])
#%%
#def batch_generator(x, y, batch_size):
#    for cbatch in range(0, x.shape[0], batch_size):
#         yield (x[cbatch:(cbatch + batch_size),:,:], [x[cbatch:(cbatch + batch_size),:,:], y_train[cbatch:(cbatch + batch_size)]])
##%%
#gen_test = batch_generator(x_train,y_train,batch_size)
#gen_val  = batch_generator(x_test, y_test,batch_size)
#for ice in gen_test:
#    print(ice)
