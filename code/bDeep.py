# %% 
# Libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras

# %% 
# Keras Libraries

from keras.models import Model
from keras.layers import  Dropout, Flatten, Dense, Conv2D, BatchNormalization,  LSTM, Input
from keras.utils.vis_utils import plot_model

# %%
#GPU Available?

# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)

# %%
# Functional API
def computationGraph():
    # CNN Input
    inputCNN = Input(shape = (72, 128, 4))

    # CONVOLUTIONAL LAYER
    conv_1 = Conv2D(filters=32, kernel_size=(8,8), strides = 4, padding = 'same', activation='relu')(inputCNN)

    # CONVOLUTIONAL LAYER 2
    conv_2 = Conv2D(filters=64, kernel_size=(4,4), strides = 2, padding = 'same', activation='relu')(conv_1)

    # CONVOLUTIONAL LAYER 3
    conv_3 = Conv2D(filters=64, kernel_size=(4,4), strides = 2, padding = 'same', activation='relu')(conv_2)

    # CONVOLUTIONAL LAYER 4
    conv_4 = Conv2D(filters=64, kernel_size=(3,3), strides = 1, padding = 'same',  activation='relu')(conv_3)

    #BatchNormalization(0.9),

    # 256 NEURONS IN DENSE HIDDEN LAYER 
    dense_1 = Dense(256, activation='relu')(conv_4)

    # 128 NEURONS IN DENSE HIDDEN LAYER 
    dense_2 = Dense(128, activation='relu')(dense_1)

    output_CNN = Flatten()(dense_2)

    # Action Graph
    inputAction = Input(shape=(8, 1))
    denseAction1 = Dense(16, activation='relu')(inputAction)
    denseAction2 = Dense(16, activation='relu')(denseAction1)
    outputAction = Dropout(0.5)(denseAction2)

    # LSTM Graph
    lstm1 = LSTM(5120, return_sequences=True)(outputAction, initial_state=[output_CNN, output_CNN])

    #Output Graph y_hat
    denseOutputY1 = Dense(16, activation='relu')(lstm1)
    denseOutputY2 = Dense(16, activation='relu')(denseOutputY1)
    y_hat = Dense(1, activation='sigmoid')(denseOutputY2)

    #Output Graph b_hat
    # denseOutputB1 = Dense(16, activation='relu')(lstm1)
    # denseOutputB2 = Dense(16, activation='relu')(denseOutputB1)
    # b_hat = Dense(1, activation='sigmoid')(denseOutputB2)

    #model = Model(inputs=[inputCNN, inputAction], outputs=[y_hat, b_hat])
    model = Model(inputs=[inputCNN, inputAction], outputs=[y_hat])

    return model

model = computationGraph()
# plot_model(model, to_file='convolutional_neural_network.png')
# print(model.summary())

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

#data = np.array([0.05, 0, -0.05, 0.05, -0.05, 0.05, 0, 0.05]).reshape((1,8,1))
#print(model.predict(data))

# %%
# Test

data = np.array([0.05, 0, -0.05, 0.05, -0.05, 0.05, 0, 0.05]).reshape((1,8,1))

img_stack=np.load('test.npy').reshape((1,72,128,4))

model([img_stack, data], training=False)

# %%
# Kahn CNN Model

# def CNN_Model():
#   CNN = Sequential([
      
#       # CONVOLUTIONAL LAYER
#       Conv2D(filters=32, kernel_size=(8,8), strides = 4, padding='same', input_shape=(64, 36, 4), activation='relu'),
      
#       # CONVOLUTIONAL LAYER 2
#       Conv2D(filters=64, kernel_size=(4,4), strides = 2,padding='same',  activation='relu'),
      
#       # CONVOLUTIONAL LAYER 3
#       Conv2D(filters=64, kernel_size=(3,3), strides = 1, padding='same',  activation='relu'),

#       #BatchNormalization(),
      
#       # 256 NEURONS IN DENSE HIDDEN LAYER 
#       Dense(256, activation='relu'),

#       # 128 NEURONS IN DENSE HIDDEN LAYER 
#       Dense(128, activation='relu'),
      
#       Flatten(),
 
#   ])

#   return CNN

# cnn = CNN_Model()
# cnn.summary()
# %%
# Manderson CNN Model

# def CNN_Model():
#   CNN = Sequential([
      
#       # CONVOLUTIONAL LAYER
#       Conv2D(filters=32, kernel_size=(8,8), strides = 4, padding='same', input_shape=(72, 128, 4), activation='relu'),
      
#       # CONVOLUTIONAL LAYER 2
#       Conv2D(filters=64, kernel_size=(4,4), strides = 2, padding='same',  activation='relu'),

#       # CONVOLUTIONAL LAYER 3
#       Conv2D(filters=64, kernel_size=(4,4), strides = 2, padding='same',   activation='relu'),
      
#       # CONVOLUTIONAL LAYER 4
#       Conv2D(filters=64, kernel_size=(3,3), strides = 1, padding='same',   activation='relu'),

#       #BatchNormalization(),
      
#       # 256 NEURONS IN DENSE HIDDEN LAYER 
#       Dense(256, activation='relu'),

#       # 128 NEURONS IN DENSE HIDDEN LAYER 
#       Dense(128, activation='relu'),
      
#       Flatten(),
 
#   ])

#   return CNN

# cnn = CNN_Model()
# cnn.summary()
# %%


