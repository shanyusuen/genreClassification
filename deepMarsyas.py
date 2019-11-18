import customRead
import numpy as np
import keras
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

batch_size = 256
epochs = 2

classes = ["Pop_Rock", "New Age", "Jazz", "RnB", "Country", "Reggae",
           "Electronic", "Folk", "Rap", "Vocal", "Latin", "Blues",
           "International"]


def transform(d):
    return classes.index(d)


model = Sequential()
model.add(BatchNormalization(axis=-1, momentum=0.99,
                             epsilon=0.001, scale=True))
model.add(Dense(100, activation='sigmoid', input_dim=124))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

x_train, y_train = customRead.read("out/all-stratified.train.arff")
x_test, y_test = customRead.read("out/all-stratified.test.arff")

x_train, x_test = np.array(x_train), np.array(x_test)

y_train = np.array(keras.utils.to_categorical([transform(y) for y in y_train], len(classes)))
y_test = np.array(keras.utils.to_categorical([transform(y) for y in y_test], len(classes)))

print(type(x_train))
print(x_train.shape)

history = keras.callbacks.callbacks.History()

_ = input("About to fit")

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

print(model.summary())
# model.save("deepMarsyas.h5")

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

input("First graphs")

test_y = np.argmax(y_test, axis=1)
pred_y = model.predict_classes(x_test)

con_mat = tf.math.confusion_matrix(labels=test_y, predictions=pred_y).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, xticklabels=classes, yticklabels=classes, annot=True, fmt='g', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()

input("Second Graph")
