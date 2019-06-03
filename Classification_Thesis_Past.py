import keras
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import h5py
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np 
#%%
# find class weight
def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss
#%%
print(K.image_data_format())
model_name = '256x256-ECTImod'
data_name ='data(256x256)-train'
filepath='weights/%s.hdf5'%(model_name)
checkpoint = ModelCheckpoint('weights/%s.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_name), monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
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
#%%
images = keras.layers.Input(input_shape)
shortcut1 = keras.layers.Conv2D(filters=256, kernel_size=(1, 1),strides=(1,1), padding="same")(images)

net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(images)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides = (2, 2), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

shortcut2 = keras.layers.Conv2D(filters=256, kernel_size=(1, 1),strides=(1,1), padding="same")(net)
net = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides = (2, 2), padding="same")(net)
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

net = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(net)
net = keras.layers.add([net,shortcut2])
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(net)
net = keras.layers.add([net,shortcut1])
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)
net = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation("relu")(net)

net = keras.layers.AveragePooling2D((256,256))(net)
net = keras.layers.Flatten()(net)

net = keras.layers.Dense(units=num_classes,activation="sigmoid")(net)

model = keras.Model(inputs=images,outputs=net)
model.summary()
#%%    
optimizer = keras.optimizers.Adadelta()
class_weights= calculating_class_weights(y_train)
model.compile(optimizer= optimizer,
              loss = get_weighted_loss(class_weights)
             )
model.summary()
print(class_weights)
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
#class_weight = (x_train.shape[0]) / (num_classes * y_train.sum(axis=0))
#class_weight = class_weight.tolist()
#%%
history = model.fit(x_train, y_train, 
          batch_size=batch_size,
          epochs=epochs,
          validation_data = (x_test, y_test),
          callbacks=callbacks_list,
          verbose=1
         )
#%%
#from keras.models import load_model
#model = load_model('weights/128x128-5-filter=16(bilinear).hdf5')
#history.history.keys()
#plt.plot(history.history['val_class_output_acc'])
#%%
#def batch_generator(x, y, batch_size):
#    for cbatch in range(0, x.shape[0], batch_size):
#         yield (x[cbatch:(cbatch + batch_size),:,:], [x[cbatch:(cbatch + batch_size),:,:], y_train[cbatch:(cbatch + batch_size)]])
##%%
#gen_test = batch_generator(x_train,y_train,batch_size)
#gen_val  = batch_generator(x_test, y_test,batch_size)
#for ice in gen_test:
#    print(ice)