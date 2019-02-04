from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split
import pickle
#%%
def Unit(x,filters,pool=False):
    res = x
    if pool:
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        res = keras.layers.Conv2D(filters=filters,kernel_size=[1,1],strides=(2,2),padding="same")(res)
    out = keras.layers.BatchNormalization()(x)
    out = keras.layers.Activation("relu")(out)
    out = keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation("relu")(out)
    out = keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = keras.layers.add([res,out])
    return out

def ResNet_MiniModel(input_shape,num_classes):
    images = keras.layers.Input(input_shape)
    net = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(images)
    net = Unit(net,32)
    net = Unit(net,32)
    net = Unit(net,32)

    net = Unit(net,64,pool=True)
    net = Unit(net,64)
    net = Unit(net,64)

    net = Unit(net,128,pool=True)
    net = Unit(net,128)
    net = Unit(net,128)

    net = Unit(net, 256,pool=True)
    net = Unit(net, 256)
    net = Unit(net, 256)
    
    net = Unit(net, 512,pool=True)
    net = Unit(net, 512)
    net = Unit(net, 512)

    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation("relu")(net)

    net = keras.layers.GlobalAveragePooling2D()(net)
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(units=num_classes,activation="sigmoid")(net)

    model = keras.Model(inputs=images,outputs=net)
    return model
# In[2]:
print(K.image_data_format())
model_name = 'log1'
data_name = 'data'

tensorboard = TensorBoard(log_dir='./logs/1', histogram_freq=0,batch_size=128)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('weights/%s.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_name), monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint,tensorboard]#,reduce_lr]
# In[3]:
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
    data = f['img'][()]
    label = f['label'][()]
#%%
model = ResNet_MiniModel(input_shape,num_classes)
optimizer = keras.optimizers.Adadelta()
loss = 'binary_crossentropy'
model.compile(optimizer= optimizer,
              loss = loss, 
              metrics = ['accuracy']        
             )
model.summary()
# In[10]:
history = model.fit(data, label, 
          batch_size=batch_size,
          epochs=epochs,
          validation_split = 0.8,
          callbacks=callbacks_list,
          verbose=1
         )
#%%
