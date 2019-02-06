import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras import backend as K
import matplotlib.pyplot as plt
import h5py
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split
#%%
print(K.image_data_format())
model_name = 'log2'
data_name = 'traintest'
filepath='weights/%s.hdf5'%(model_name)

tensorboard = TensorBoard(log_dir='./logs/1', histogram_freq=0,batch_size=128)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('weights/%s.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_name), monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint,tensorboard]#,reduce_lr]
#%%
batch_size = 1
num_classes = 17
epochs = 500
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
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
#%%
base_model = InceptionResNetV2(weights=None, include_top=False,input_shape=(256,256,3))
x = base_model.output
x = AveragePooling2D((6,6))(x)
x = Flatten()(x)
predictions = Dense(17, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)    
#%%    
optimizer = keras.optimizers.Adadelta()
loss = 'binary_crossentropy'
model.compile(optimizer= optimizer,
              loss = loss, 
              metrics = ['accuracy']        
             )
model.summary()
# In[10]:
history = model.fit(x_train, y_train, 
          batch_size=batch_size,
          epochs=epochs,
          validation_data = (x_test, y_test),
          callbacks=callbacks_list,
          verbose=1
         )
#%% save and load part
#model_json = model.to_json()
name = '%s_last'%(model_name)
## save model and weight
model.save('weights/%s.hdf5'%(name))
