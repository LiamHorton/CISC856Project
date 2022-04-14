# %% Libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras

# %% Keras Libraries

from keras.models import Sequential, Model
from keras.layers import  Dropout, Flatten, Dense, Conv2D, BatchNormalization,  MaxPool2D,  LSTM, TimeDistributed, Input
from keras.utils.vis_utils import plot_model


# %%

# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)

# %%

# Functional API
# def CNN_Model():
#   CNN = Sequential([
      
#       # CONVOLUTIONAL LAYER
#       Conv2D(filters=32, kernel_size=(8,8), strides = 4, input_shape=(227, 227, 12), activation='relu'),
      
#       # CONVOLUTIONAL LAYER 2
#       Conv2D(filters=64, kernel_size=(4,4), strides = 2, activation='relu'),

#       # CONVOLUTIONAL LAYER 3
#       Conv2D(filters=64, kernel_size=(4,4), strides = (2,2), activation='relu'),
      
#       # CONVOLUTIONAL LAYER 4
#       Conv2D(filters=64, kernel_size=(3,3), strides = 1, activation='relu'),

#       BatchNormalization(),
      
#       # 256 NEURONS IN DENSE HIDDEN LAYER 
#       Dense(256, activation='relu'),

#       # 128 NEURONS IN DENSE HIDDEN LAYER 
#       Dense(128, activation='relu'),
      
#       Flatten(), 
#   ])

#   return CNN

# cnn = CNN_Model()
# cnn.summary()

# vector = cnn(x, training=False)

actions = np.zeros(8)

# lstm, h = LSTM_Model()
# lstm.summary()
#vector = lstm(actions, training=False)

# inputs = Input(shape=(227, 227, 12))
# inputs2 = Input(shape = (actions.shape[0], 1))

# model = TimeDistributed(CNN_model)(inputs)
# lstm, h = LSTM(8, activation = 'relu', return_sequences=True)(inputs2)
# final_model = Model(inputs=inputs2, outputs=[lstm, h])
# model = Dense(1, activation='sigmoid')(model)

inputs1 = Input(shape=(8, 1))
lstm1 = LSTM(1, return_sequences=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1])


data = np.array([0, 0.05, -0.05, 0.05, -0.05, 0, 0, 0]).reshape((1,8,1))

print(model.predict(data))
#final_model = Model(inputs=inputs, outputs=model)

#Dense(1, activation='sigmoid'),
#print(model.summary())
#plot_model(model, to_file='convolutional_neural_network.png')
#plot_model(model, to_file='convolutional_neural_network.png')
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
#CNN_model.evaluate(data, verbose = 1)
#CNN_model.fit([data], labels, epochs = 3)

# %%
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# %%
print(model.summary())
plot_model(model, to_file='convolutional_neural_network.png')
# %%


# def build_LSTM_CNN_net()
#       from keras.applications.vgg16 import VGG16
#       from keras.models import Model
#       from keras.layers import Dense, Input, Flatten
#       from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D
#       from keras.layers.recurrent import LSTM
#       from keras.layers.wrappers import TimeDistributed
#       from keras.optimizers import Nadam
    
    
#       from keras.applications.vgg16 import VGG16

#       num_classes = 5
#       frames = Input(shape=(5, 224, 224, 3))
#       base_in = Input(shape=(224,224,3))
    
#       base_model = VGG16(weights='imagenet',
#                   include_top=False,
#                   input_shape=(224,224,3))
    
#       x = Flatten()(base_model.output)
#       x = Dense(128, activation='relu')(x)
#       x = TimeDistributed(Flatten())(x)
#       x = LSTM(units = 256, return_sequences=False, dropout=0.2)(x)
#       x = Dense(self.nb_classes, activation='softmax')(x)
    
# lstm_cnn = build_LSTM_CNN_net()
# keras.utils.plot_model(lstm_cnn, "lstm_cnn.png", show_shapes=True)