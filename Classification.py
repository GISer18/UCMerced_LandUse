import keras
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, Flatten, Conv2D
from keras import backend as K
import matplotlib.pyplot as plt
import h5py
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
#%%
print(K.image_data_format())
model_name = 'logv.2_1'
data_name = 'traintest2'
filepath='weights/%s.hdf5'%(model_name)

tensorboard = TensorBoard(log_dir='./logs/v.2_1', histogram_freq=0,batch_size=128)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('weights/%s.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_name), monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint,tensorboard]#,reduce_lr]
#%%
batch_size = 8
num_classes = 17
epochs = 100
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
y_ints = [y.argmax() for y in y_train]
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)
#%% simple model
from keras.applications.vgg16 import VGG16
base_model = VGG16(weights=None, include_top=False,input_shape=(256,256,3))
base_model.summary()
#%%    
x = base_model.get_layer('block3_conv3').output
x = Conv2D(256,kernel_size=(3,3),padding='same')(x)
x = keras.layers.Activation("relu")(x)
x = AveragePooling2D((64,64))(x)
x = Flatten()(x)
predictions = Dense(17, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions) 
# In[10]:
optimizer = keras.optimizers.Adadelta()
model.compile(optimizer= optimizer,loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
)
x_test = (x_test/255.0).astype(np.float32)
#%%
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              validation_data = (x_test,y_test),
                              steps_per_epoch=len(x_train) // batch_size,
                              class_weight=class_weights,
                              epochs=200,
                              verbose=1,
                              callbacks = callbacks_list
                              )
import pickle
with open('weights/%s.pickle'%(model_name), 'wb') as file_pi:
    pickle.dump(history.history, file_pi) 
#%%    
#%%
preds = model.predict(x_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
#%%
import matplotlib
matplotlib.style.use('seaborn')
epochs = len(history.history['loss'])
max_loss = max(max(history.history['loss']), max(history.history['val_loss']))
plt.axis([0, epochs+1, 0, round(max_loss * 2.0) / 2 + 0.5])
x = np.arange(1, epochs+1)
plt.plot(x, history.history['loss'])
plt.plot(x, history.history['val_loss'])
plt.title('Training loss vs. Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='right')
plt.show()
#%%
epochs = np.argmin(history.history['val_loss']) + 1
print(f'Stop training at {epochs} epochs')
#%%





























