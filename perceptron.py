import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
import matplotlib.pylab as plt
import dataRead

batch_size = 256
num_classes = 13
epochs = 10

dimension = 60

data_set = dataRead.load_data(dataRead.data_file)

classes = ["'Pop_Rock'", "'New Age'", "'Jazz'", "'RnB'", "'Country'", "'Reggae'", "'Electronic'", "'Folk'", "'Rap'", "'Vocal'", "'Latin'", "'Blues'", "'International'"]
transform = lambda x: classes.index(x)

x_train = data_set[0:360000, 0:60]
y_train = data_set[0:360000, 60]

x_test = data_set[360000:, 0:60]
y_test = data_set[360000:, 60]

y_train = keras.utils.to_categorical([transform(x) for x in y_train], num_classes)
y_test = keras.utils.to_categorical([transform(x) for x in y_test], num_classes)

model = Sequential()
model.add(Dense(60, input_dim=60, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(13, activation='softmax'))

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
