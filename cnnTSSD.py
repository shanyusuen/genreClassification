import customRead
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Flatten, MaxPooling3D

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

batch_size = 2048
epochs = 50

categories = ["Pop_Rock", "New Age", "Jazz", "RnB", "Country",
              "Reggae", "Electronic", "Folk", "Rap", "Vocal", "Latin",
              "Blues", "International"]


def transform(v):
    return categories.index(v)


x_train, y_train = customRead.read("out/all-stratified.train.arff")
"""
x_train = np.array([np.reshape(x_train[i], (1176//7, 7, 1))
                    for i in range(len(x_train))])
"""

x_train = np.array([np.reshape(x_train[i], (7, 7, 24, 1))
                    for i in range(len(x_train))])

print("Transformed")

x_test, y_test = customRead.read("out/all-stratified.test.arff")
# x_test = x_test[0:1000]
# y_test = y_test[0:1000]
"""
x_test = np.array([np.reshape(x_train[i], (1176//7, 7, 1))
                   for i in range(len(x_test))])
"""

x_test = np.array([np.reshape(x_train[i], (7, 7, 24, 1))
                   for i in range(len(x_test))])

y_train = keras.utils.to_categorical([transform(x) for x in y_train],
                                     len(categories))

y_test = keras.utils.to_categorical([transform(x) for x in y_test],
                                    len(categories))

print("Number of each label (train):\n", np.sum(y_train, axis=0))
print("Number of each label (test):\n", np.sum(y_test, axis=0))


model = Sequential()
# original kernel: (5, 3)
# activation='relu',
model.add(Conv3D(16, kernel_size=(3, 3, 7), strides=(1, 1, 1),
                 data_format='channels_last',
                 kernel_initializer='glorot_uniform',
                 input_shape=(7, 7, 24, 1)))
# original stride: (4, 1)
model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=(1, 1, 2),
                       data_format="channels_first"))
# original kernel: (5,3)
model.add(Conv3D(32, (3, 3, 7)))
# , activation='relu'
model.add(MaxPooling3D(pool_size=(1, 1, 1)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def __init__(self):
        self.acc = list()

    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('accuracy'))


history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
print(history.acc)
