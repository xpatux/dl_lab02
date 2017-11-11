#Definicion de librerias con la funciones que seran utilizadas por Keras.
import keras
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

#Definicion de contenedor y primera capa de AlexNet.
modelAlexNet = Sequential()
modelAlexNet.add(ZeroPadding2D((2,2), input_shape=(224, 224, 3)))

# modelAlexNet.add(Convolution2D(96,11,11,subsample=(4,4),border_mode='valid'))
# /home/diego/anaconda3/envs/tensor/lib/python3.6/site-packages/ipykernel/__main__.py:4: 
# UserWarning: Update your `Conv2D` call to the Keras 2 API: 
# `Conv2D(96, (11, 11), strides=(4, 4), padding="valid")`

modelAlexNet.add(Conv2D(96, (11,11), strides=(4,4), padding='valid'))

modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(BatchNormalization())
print(modelAlexNet.output_shape)
modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))
print(modelAlexNet.output_shape)
