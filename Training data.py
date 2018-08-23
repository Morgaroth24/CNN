import sys
import os

from tensorflow.python.keras.prepocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models  import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clar_session()
data_training = './data/training'
data_validation = './data/validation'

## Parameters
epochs = 20
height, lenght = 100, 100
batch_size = 32
steps = 1000
steps_validation = 200
filters_Conv1 = 32
filters_Conv2 = 64
filter_size1= (3,3)
filter_size2= (2,2)
pool_size = (2,2)
class_size = 3    ##Numero de clases
lr = 0.0005

## Preprosesing the images

trainning_datagen = ImageDataGenerator(
    rescale=1./255, ## Cada uno de los pixeles seran rescalados de 0 a 1
    shear_range=0.3, ## Las imagenes las inclina
    zoom_range=0.3,  ## Las imagenes les hara zoom 
    horizontal_flip=True ##Las imagenes las invierte 
    )

validation_datagen = ImageDataGenerator(
    rescale=1./255
    )

training_image = training_datagen.flow_from_directory(
    data_training,
    target_size=(height, lenght),
    batch_size=batch_size,
    class_mode='categorical'
    )

validation_image = validation_datagen.flow_from_directory(
    data_validation,
    target_size=(height, lenght),
    batch_size=batch_size,
    class_mode='categorical'
    )


## CNN

cnn = Sequential()
cnn.add(Convolution2D(filters_Conv1,
                      filter_size1,
                      padding='same',
                      input_shape=(height,lenght,3),
                      activation='relu')
        )
cnn.add(MaxPooling2D(pool_size=pool_size))
cnn.add(Convolution2D(filters_Conv2,
                      filter_size2,
                      padding='same',
                      activation='relu')
        )
cnn.add(MaxPooling2D(pool_size=pool_size))
cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(class_size,activation='softmax'))

cnn.compile(loss='cattegorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

cnn.fit_generatopr(training_image,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=validation_image,
        validation_steps=steps_validation)

dir='./modelo/'
if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
