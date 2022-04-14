# %% 
# Libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras

# %% 
# Keras Libraries

from keras.models import Sequential, Model
from keras.layers import  Dropout, Flatten, Dense, Conv2D, BatchNormalization,  MaxPool2D,  LSTM, TimeDistributed, Input
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

# vector = cnn(x, training=False)

actions = np.zeros(8)

# lstm, h = LSTM_Model()
# lstm.summary()
#vector = lstm(actions, training=False)

#inputs = Input(shape=(227, 227, 12))
# inputs2 = Input(shape = (actions.shape[0], 1))

# model = TimeDistributed(CNN_model)(inputs)
# lstm, h = LSTM(8, activation = 'relu', return_sequences=True)(inputs2)
# final_model = Model(inputs=inputs2, outputs=[lstm, h])
# model = Dense(1, activation='sigmoid')(model)


input_CNN = Input(shape = (78, 128, 12))

# CONVOLUTIONAL LAYER
conv_1 = Conv2D(filters=32, kernel_size=(8,8), strides = 4, activation='relu')(input_CNN)

# CONVOLUTIONAL LAYER 2
conv_2 = Conv2D(filters=64, kernel_size=(4,4), strides = 2, activation='relu')(conv_1)

# CONVOLUTIONAL LAYER 3
conv_3 = Conv2D(filters=64, kernel_size=(4,4), strides = 2, activation='relu')(conv_2)

# CONVOLUTIONAL LAYER 4
conv_4 = Conv2D(filters=64, kernel_size=(3,3), strides = 1, activation='relu')(conv_3)

#BatchNormalization(),

# 256 NEURONS IN DENSE HIDDEN LAYER 
dense_1 = Dense(256, activation='relu')(conv_4)

# 128 NEURONS IN DENSE HIDDEN LAYER 
dense_2 = Dense(128, activation='relu')(dense_1)

output_cnn = Flatten()(dense_2)

# input_action = Input(shape=(1, 1))

# dense_1 = Dense(256, activation='relu')(input_action)
# dense_2 = Dense(512, activation='relu')(dense_1)
# dropout = Dropout(0.5)(dense_2)
# output_action = Flatten()(dropout)

input_LSTM = Input(shape=(8, 1))
lstm1 = LSTM(512, return_sequences=True)(input_LSTM, initial_state=[output_cnn, output_cnn])
#lstm1 = LSTM(1, return_state=True)
#print(lstm1.states)
model = Model(inputs=[input_CNN, input_LSTM], outputs=lstm1)
plot_model(model, to_file='convolutional_neural_network.png')

#data = np.array([0.05, 0, -0.05, 0.05, -0.05, 0, 0, 0.05]).reshape((1,8,1))


#CNN_model = Model(inputs=input_CNN, outputs=output)
#print(CNN_model.summary())


#initial_state = Flatten()
#data = np.array([0.05, 0, -0.05, 0.05, -0.05, 0, 0, 0.05]).reshape((1,8,1))


# lstm1 = LSTM(1, return_sequences=True)(input_LSTM, initial_state=dense_2)
# lstm1 = LSTM(1, return_state=True)
# output, hidden, cell = lstm1(data)
# #print(lstm1.states)
# model = Model(inputs=input_LSTM, outputs=lstm1)

#lstm1, state_h, state_c = LSTM(1, return_sequences=True)(input_LSTM)
#model = Model(inputs=input_LSTM, outputs=[lstm1, state_h, state_c])
# define input data
#data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
#print(model.predict(data))

#print(model.predict(data))
#final_model = Model(inputs=inputs, outputs=model)

#Dense(1, activation='sigmoid'),
#print(model.summary())
#plot_model(model, to_file='convolutional_neural_network.png')
#plot_model(model, to_file='convolutional_neural_network.png')
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
#CNN_model.evaluate(data, verbose = 1)
#CNN_model.fit([data], labels, epochs = 3)

# %%

input_action = Input(shape=(1, 1))

dense_1 = Dense(256, activation='relu')(input_action)
dense_2 = Dense(512, activation='relu')(dense_1)
dropout = Dropout(0.5)(dense_2)
output = Flatten()(dropout)

action_model = Model(inputs=input_action, outputs=output)
print(action_model.summary())

# %%
input_LSTM = Input(shape=(512, 1))
#data = np.array([0.05, 0, -0.05, 0.05, -0.05, 0, 0, 0.05]).reshape((1,8,1))
lstm1 = LSTM(1, return_sequences=True)(input_LSTM)
#lstm1 = LSTM(1, return_state=True)
#print(lstm1.states)
model = Model(inputs=action_model.output, outputs=lstm1)

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

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

# %%

def CNN_Model():
  CNN = Sequential([
      
      # CONVOLUTIONAL LAYER
      Conv2D(filters=32, kernel_size=(8,8), strides = 4, input_shape=(78, 128, 12), activation='relu'),
      
      # CONVOLUTIONAL LAYER 2
      Conv2D(filters=64, kernel_size=(4,4), strides = 2, activation='relu'),

      # CONVOLUTIONAL LAYER 3
      Conv2D(filters=64, kernel_size=(4,4), strides = 2, activation='relu'),
      
      # CONVOLUTIONAL LAYER 4
      Conv2D(filters=64, kernel_size=(3,3), strides = 1, activation='relu'),

      #BatchNormalization(),
      
      # 256 NEURONS IN DENSE HIDDEN LAYER 
      Dense(256, activation='relu'),

      # 128 NEURONS IN DENSE HIDDEN LAYER 
      Dense(128, activation='relu'),
      
      Flatten(),
 
  ])

  return CNN

cnn = CNN_Model()
cnn.summary()
# %%
