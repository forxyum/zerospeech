import encoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Input, Add, Activation, Dense, UpSampling2D

#to get the last layer and input shape variables
encoder.get_conv_model()

#TODO: jitter

x = Conv1D(128,3,padding='same',activation='relu')(encoder.last_layer)
x = UpSampling2D(320)(x)

model = Model(inputs=encoder.inputs,outputs=x)
model.summary()

