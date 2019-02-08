import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten, Conv2D
from keras import backend as K
import matplotlib.pyplot as plt
import h5py
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import pickle
#%% find class weight
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
model_name = 'log4'
data_name = 'traintest'
filepath='weights/%s.hdf5'%(model_name)

tensorboard = TensorBoard(log_dir='./logs/4', histogram_freq=0,batch_size=128)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('weights/%s.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_name), monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint,tensorboard]#,reduce_lr]
#%%
batch_size = 32
num_classes = 17
epochs = 200
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
    x_val = f['x_val'][()]
    y_val = f['y_val'][()]
#%% simple model
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5),padding='same', activation='relu', input_shape=input_shape) ,
    keras.layers.Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu') ,
    keras.layers.MaxPooling2D(pool_size=(2, 2)) ,
    
    keras.layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu') ,
    keras.layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu') ,
    keras.layers.MaxPooling2D(pool_size=(2, 2)) ,
    keras.layers.Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu') ,
    keras.layers.AveragePooling2D(pool_size=(64,64)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(num_classes,activation='sigmoid')
    ])
#%%    
optimizer = keras.optimizers.Adam(lr=0.00125)
#optimizer = keras.optimizers.SGD(lr=0.1,momentum=0.9,decay = 1e-4)
#optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


#loss = 'binary_crossentropy'
class_weights= calculating_class_weights(y_train)
model.compile(optimizer= optimizer,
              loss = get_weighted_loss(class_weights)
             )
model.summary()
# In[10]:
datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
#   ,preprocessing_function = preprocess_input
    )  
x_test = x_test/255.0
#%%
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data = (x_test,y_test),  
                    steps_per_epoch=2*len(x_train) // batch_size, 
                    epochs=100,
                    verbose=1,
                    callbacks = callbacks_list
                    )
with open('weights/%s.pickle'%(model_name), 'wb') as file_pi:
    pickle.dump(history.history, file_pi) 
#%%    
#with open('weights/%s.pickle'%(model_name), 'rb') as input_file:
#    e = pickle.load(input_file)
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





























