import numpy as np
import tensorflow as tf
from tensorflow import keras

# %%
# Keras Libraries

# from keras.models import Model
# from keras.layers import  Dropout, Flatten, Dense, Conv2D, BatchNormalization, LSTM, Input
# from keras.utils.vis_utils import plot_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dropout, Flatten, Dense, Conv2D, BatchNormalization, LSTM, Input

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

def computation_graph(H, img_x=128, img_y=72):
    # CNN Input
    inputCNN = Input(shape=(img_y, img_x, 4))

    # CONVOLUTIONAL LAYER
    conv_1 = Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='same', activation='relu')(inputCNN)

    # CONVOLUTIONAL LAYER 2
    conv_2 = Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same', activation='relu')(conv_1)

    # CONVOLUTIONAL LAYER 3
    conv_3 = Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same', activation='relu')(conv_2)

    # CONVOLUTIONAL LAYER 4
    conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv_3)

    # BatchNormalization(0.9),

    # 256 NEURONS IN DENSE HIDDEN LAYER
    dense_1 = Dense(256, activation='relu')(conv_4)

    # 128 NEURONS IN DENSE HIDDEN LAYER
    dense_2 = Dense(128, activation='relu')(dense_1)

    output_CNN = Flatten()(dense_2)

    # Action Graph
    inputAction = Input(shape=(H, 1))
    denseAction1 = Dense(16, activation='relu')(inputAction)
    denseAction2 = Dense(16, activation='relu')(denseAction1)
    outputAction = Dropout(0.5)(denseAction2)

    # LSTM Graph
    lstm1 = LSTM(output_CNN.shape[1], return_sequences=True)(outputAction, initial_state=[output_CNN, output_CNN])

    # Output Graph y_hat
    denseOutputY1 = Dense(16, activation='relu')(lstm1)
    denseOutputY2 = Dense(16, activation='relu')(denseOutputY1)
    y_hat = Dense(1, activation='sigmoid')(denseOutputY2)

    # Output Graph b_hat
    # denseOutputB1 = Dense(16, activation='relu')(lstm1)
    # denseOutputB2 = Dense(16, activation='relu')(denseOutputB1)
    # b_hat = Dense(1, activation='sigmoid')(denseOutputB2)

    # model = Model(inputs=[inputCNN, inputAction], outputs=[y_hat, b_hat])
    model = Model(inputs=[inputCNN, inputAction], outputs=[y_hat])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def run(model, img_stack, actions):
    actions = np.array(actions).reshape((1, actions.size, 1))
    img_stack = np.load('test.npy').reshape((1, img_stack.shape[0], img_stack.shape[1], img_stack.shape[2]))
    ys = model([img_stack, actions], training=False)
    ys = np.array(ys)[:,:,0]
    return ys

def train(model, data_I, data_a, y_labels):
    data_I = np.stack(data_I)
    data_a = np.stack(data_a)
    y_labels = np.stack(y_labels)
    model.fit([data_I, data_a], y_labels)