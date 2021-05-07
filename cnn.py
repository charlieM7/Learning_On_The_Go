import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from data import Data


def cnn():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation=tf.nn.softmax))

    return model


def train(data, network):
    index = 0;
    while index != 60:
        train_images, train_labels = data.train(index)
        network.fit(x=train_images, y=train_labels, epochs=3)
        index = index + 1


def test(data, network):
    index = 0
    total = 0
    unknown = 0
    unlabeled = []

    while index != 10:
        test_images, test_labels = data.test(index)
        index = index + 1
        prediction = network.predict(test_images)

        for i in range(len(test_images)):
            value = np.max(prediction[i])
            total = total + 1

            if value <= 0.7:
                unknown = unknown + 1
                unlabeled.append((test_labels[i], test_images[i]))

    accuracy = unknown/total
    return accuracy, unlabeled


if __name__ == '__main__':
    data = Data()
    network = cnn()

    network.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    train(data, network)
    accuracy, unlabeled = test(data, network)
    print(accuracy)


