import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
import matplotlib.pylab as plt
import dataRead
import data_plotting


test_file = ".\\out\\test.arff"
train_file = ".\\out\\train.arff"
classes = ["Pop_Rock", "New_Age", "Jazz", "RnB", "Country", "Reggae", "Electronic", "Folk", "Rap", "Vocal", "Latin", "Blues", "International"]


batch_size = 256
epochs = 10



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

data_plotting.plotConfusion(model, classes, x_test, y_test)
data_plotting.plotHistory(test_accuracy, training_accuracy, test_loss, training_loss)



