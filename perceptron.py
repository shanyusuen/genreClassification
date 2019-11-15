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


class AccuracyHistory(keras.callbacks.Callback):
    def __init__(self):
        self.acc = []

    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print("Accuracies: ", history.acc)
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
