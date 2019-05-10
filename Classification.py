import keras
from keras import backend as K
import h5py
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
#%% set parameters
print(K.image_data_format())
model_name = 'Normal-Approach'
data_name = 'data(8x8)'
#%%
batch_size = 256
num_classes = 4
epochs = 100    
img_size = 8
input_shape = (img_size, img_size, 3)
#%% load data
hf = h5py.File('dataset/%s.h5'%(data_name), 'r')
print(list(hf.keys()))
hf.close()
#%%
with h5py.File('dataset/%s.h5'%(data_name), 'r') as f:
    img = f['img'][()]
    label = f['label'][()]
    groundtruth = f['groundtruth'][()] 
label = keras.utils.to_categorical(label, num_classes=num_classes, dtype='int8')
#%% build the model
train_size = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
for size in train_size:
    print("Train_size: ",size)
    checkpoint = ModelCheckpoint('weights/%s_train_size=%.2f.hdf5'%(model_name,size), monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
    tensorboard = TensorBoard(log_dir='./logs/%s_train_size=%.2f.hdf5'%(model_name,size) , histogram_freq=0,  write_graph=True, write_images=False)
    callbacks_list = [checkpoint,tensorboard,earlystop]
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3,3),padding='same', activation='relu', input_shape=input_shape) ,
        keras.layers.Conv2D(64, kernel_size=(3,3),padding='same', activation='relu') ,
        keras.layers.MaxPooling2D(pool_size=(2, 2)) ,
        keras.layers.Dropout(0.25) ,
        
        keras.layers.Conv2D(128, kernel_size=(3,3),padding='same', activation='relu') ,
        keras.layers.Conv2D(128, kernel_size=(3,3),padding='same', activation='relu') ,
        keras.layers.MaxPooling2D(pool_size=(2, 2)) ,
        keras.layers.Dropout(0.25) , 
    
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(num_classes,activation='softmax')
        ])
    optimizer = keras.optimizers.Adadelta()
    model.compile(optimizer= optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                 )
    #%
    x_train, x_test, y_train, y_test, groundtruth_train, groundtruth_test = train_test_split(img, label, groundtruth, train_size=size, test_size=1-size, random_state=42)
    
    hf = h5py.File('dataset/%s_train_size=%.2f.h5'%(model_name,size), 'w')
    hf.create_dataset('x_test',data=x_test)
    hf.create_dataset('y_test',data=y_test)
    hf.create_dataset('groundtruth_test',data=groundtruth_test)
    hf.close()
    
    x_train = x_train/255.0
    x_test = x_test/255.0

    model.fit(x_train, y_train, 
            batch_size=batch_size,
            epochs=epochs,
            validation_data = (x_test, y_test),
            callbacks=callbacks_list,
            verbose=2
            )
    keras.backend.clear_session()
#%%
#from keras.models import load_model
#model = load_model('weights/128x128-5-filter=16(bilinear).hdf5')