import customRead
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import MaxPooling2D, BatchNormalization

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

batch_size = 512
epochs = 50

categories = ["'Pop_Rock'", "'New Age'", "'Jazz'", "'RnB'", "'Country'",
              "'Reggae'", "'Electronic'", "'Folk'", "'Rap'", "'Vocal'", "'Latin'",
              "'Blues'", "'International'"]


def transform(v):
    return categories.index(v)


model = Sequential()
model.add(BatchNormalization(axis=-1, momentum=0.99,
                             epsilon=0.001, scale=True))
model.add(Conv2D(8, kernel_size=(3, 7), strides=(1, 1),
                 data_format='channels_last',
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 input_shape=(7, 24, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                       data_format="channels_last"))
model.add(Conv2D(16, (3, 5), padding='same', data_format='channels_last'))
# model.add(Conv2D(512, (3, 3), padding='same', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(512, (2, 2), padding='same', data_format='channels_last'))
# model.add(Conv2D(512, (2, 2), padding='same', data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 3)))
model.add(Flatten())
# model.add(Dropout(0.1))
# model.add(BatchNormalization(axis=-1, momentum=0.99,
# epsilon=0.001, scale=True))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

x_train, y_train = customRead.read("out/partition.train.arff")
"""
x_train = np.array([np.reshape(x_train[i], (1176//7, 7, 1))
                    for i in range(len(x_train))])
"""

x_train = np.array([np.reshape(x_train[i], (7, 24, 1))
                    for i in range(len(x_train))])

x_test, y_test = customRead.read("out/partition.test.arff")
# x_test = x_test[0:1000]
# y_test = y_test[0:1000]
"""
x_test = np.array([np.reshape(x_train[i], (1176//7, 7, 1))
                   for i in range(len(x_test))])
"""

x_test = np.array([np.reshape(x_test[i], (7, 24, 1))
                   for i in range(len(x_test))])

y_train = keras.utils.to_categorical([transform(x) for x in y_train],
                                     len(categories))

y_test = keras.utils.to_categorical([transform(x) for x in y_test],
                                    len(categories))


history = keras.callbacks.callbacks.History()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

print(model.summary())

yellow = "#F4D03F"
blue = "#5DADE2"

training_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']
training_loss = history.history['loss']
test_loss = history.history['val_loss']

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(7, 7))

ax0.plot(range(1, 1 + len(test_accuracy)), training_accuracy, color=blue)
ax0.set_ylabel('Accuracy')
ax0.set_title("Training Data")

ax1.plot(range(1, 1 + len(test_accuracy)), test_accuracy, color=blue)
ax1.set_xlabel('Epochs')
ax1.set_title("Testing Data")

ax2.plot(range(1, 1 + len(training_loss)), training_loss, color=yellow)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')

ax3.plot(range(1, 1 + len(test_loss)), test_loss, color=yellow)
ax3.set_xlabel('Epochs')

plt.tight_layout()
plt.show()

test_y = np.argmax(y_test, axis=1)
pred_y = model.predict_classes(x_test)

con_mat = tf.math.confusion_matrix(labels=test_y, predictions=pred_y).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, xticklabels=categories, yticklabels=categories, annot=True, fmt='g', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()

input("End graphs")
