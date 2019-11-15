import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
import matplotlib.pylab as plt
import dataRead


test_file = ".\\out\\test.arff"
train_file = ".\\out\\train.arff"
classes = ["Pop_Rock", "New_Age", "Jazz", "RnB", "Country", "Reggae", "Electronic", "Folk", "Rap", "Vocal", "Latin", "Blues", "International"]


batch_size = 256
epochs = 100



test_data = dataRead.load_data(test_file)
train_data = dataRead.load_data(train_file)



#last dimension is label
num_dimensions = test_data.shape[1] - 1
num_classes = len(classes)

transform = lambda x: classes.index(x)

x_train = test_data[:, :-1]
y_train = test_data[:, -1:]

x_test = test_data[:, :-1]
y_test = test_data[:, -1:]

y_train = keras.utils.to_categorical([transform(x) for x in y_train], num_classes)
y_test = keras.utils.to_categorical([transform(x) for x in y_test], num_classes)

model = Sequential()
model.add(Dense(num_dimensions, input_dim=num_dimensions, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


def plotHistory(test_accuracy, training_accuracy, test_loss, training_loss):

    yellow  = "#f4d03f"
    blue = "#5dade2"

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

    # Accuracy

    ax0.plot(range(1, 1 + len(training_accuracy)), training_accuracy, color=blue)
    ax0.set_ylabel('Accuracy')
    ax0.set_title("Training Data")

    ax1.plot(range(1, 1 + len(test_accuracy)), test_accuracy, color=blue)
    ax1.set_xlabel('Epochs')
    ax1.set_title("Testing Data")

    # Loss

    ax2.plot(range(1, 1 + len(training_loss)), training_loss, color=yellow)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    ax3.plot(range(1, 1 + len(test_loss)), test_loss, color=yellow)
    ax3.set_xlabel('Epochs')

    plt.show()

import tensorflow as tf
import numpy as np
#import pandas as pd
import seaborn as sns

def plotConfusion(model, classes, test_x, test_y):
    # test labels can not be one hot encoded
    test_y = np.argmax(test_y, 1)

    y_pred = model.predict_classes(test_x)
    con_mat = tf.math.confusion_matrix(labels=test_y, predictions=y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    """
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)
    """
    figure = plt.figure(figsize=(8, 8))
    #sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    sns.heatmap(con_mat, xticklabels=classes, yticklabels=classes, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

callback = keras.callbacks.callbacks.History()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[callback])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
training_accuracy = callback.history['accuracy']
test_accuracy = callback.history['val_accuracy']
training_loss = callback.history['loss']
test_loss = callback.history['val_loss']

plotConfusion(model, classes, x_test, y_test)
plotHistory(test_accuracy, training_accuracy, test_loss, training_loss)



