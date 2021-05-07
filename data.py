import random
import numpy as np


def shuffle(set_images, set_labels):

    shuffler = np.random.permutation(len(set_images))

    set_images_shuffled = set_images[shuffler]
    set_labels_shuffled = set_labels[shuffler]

    return set_images_shuffled, set_labels_shuffled


def rotate(set_images):
    for image in set_images:
        index = random.randint(0, 3)

        np.rot90(image, k=index)

    return set_images


class Data:

    def __init__(self):
        self.train_data = []
        self.test_data = []

        with open("Data/mnist/mnist_train.csv", "r") as file:
            for line in file:
                self.train_data.append(line)
        file.close()

        with open("Data/mnist/mnist_test.csv", "r") as file:
            for line in file:
                self.test_data.append(line)
        file.close()

    def train(self, index):
        i = index * 1000

        train_labels = []
        train_images = []

        for line in self.train_data[i:i + 1000]:
            line_split = line.split(",")
            train_labels.append(line_split[0])
            image = np.asfarray(line_split[1:], dtype='float32')
            image_2d = image.reshape(28, 28, 1)
            image_2d /= 255
            train_images.append(image_2d)

        train_labels, train_images = np.array(train_labels).astype('float32'), np.array(train_images).astype('float32')
        train_filter = np.isin(train_labels, [0, 1, 2, 3, 4])
        train_images, train_labels = train_images[train_filter], train_labels[train_filter]

        train_images = rotate(train_images)
        train_images, train_labels = shuffle(train_images, train_labels)
        return train_images, train_labels

    def test(self, index):
        i = index * 1000

        test_labels = []
        test_images = []

        for line in self.test_data[i:i + 1000]:
            line_split = line.split(",")
            test_labels.append(line_split[0])
            image = np.asfarray(line_split[1:], dtype='float32')
            image_2d = image.reshape(28, 28, 1)
            image_2d /= 255
            test_images.append(np.array(image_2d))

        test_labels, test_images = np.array(test_labels).astype('float32'), np.array(test_images).astype('float32')
        test_filter = np.isin(test_labels, [5, 6, 7, 8, 9])
        test_images, test_labels = test_images[test_filter], test_labels[test_filter]

        test_images = rotate(test_images)
        test_images, test_labels = shuffle(test_images, test_labels)

        return test_images, test_labels
