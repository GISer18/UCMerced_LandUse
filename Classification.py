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
checkpoint = ModelCheckpoint('weights/%s.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_name), monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint,tensorboard]#,reduce_lr]
#%%
batch_size = 16
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
#%% inceptionresnetv2
base_model = InceptionResNetV2(weights='imagenet', include_top=False,input_shape=(256,256,3))
x = base_model.get_layer('block17_20_ac').output
#x = keras.layers.Conv2D(1088, kernel_size=(3, 3),padding='same', activation='relu')(x)
x = AveragePooling2D((14,14))(x)
x = Flatten()(x)
predictions = Dense(17, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)    

for layer in model.layers[:-19]:
    layer.trainable = False

for layer in model.layers:
    print(layer, layer.trainable)

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
optimizer = keras.optimizers.Adam(lr=0.0001)
#loss = 'binary_crossentropy'
class_weights= calculating_class_weights(y_train)
model.compile(optimizer= optimizer,
              loss = get_weighted_loss(class_weights),        
             )
model.summary()
# In[10]:
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
#   ,preprocessing_function = preprocess_input
    )  
#%%
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data = (x_test,y_test),  
                    steps_per_epoch=len(x_train) // batch_size, 
                    epochs=100
#                    ,callbacks = callbacks_list
                    )
#%% save and load part
#model_json = model.to_json()
name = '%s_last'%(model_name)
## save model and weight
model.save('weights/%s.hdf5'%(name))

#%%

preds = model.predict(x_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
#%%

#np.all(preds==y_test,axis=1)print(y_test[i])
#i=10
#
#print(preds[i].astype(np.uint16))






























