# Learning_On_The_Go

# Description 
(work in progress)   
The goal of this project was to create a network that is able to learn on the go, meaning that it would be able to classify data it has never been trained on. In order to accomplish this, the first step is to identify items that belong to unseen categories. Using the Modified National Institute of Standards and Technology (MNIST) dataset we train a convolutional neural network (CNN) on digits 0-4. Later, we test the network on digits 5-9 to see if it correctly identifies the unseen numbers.

# Set Up

## Download

1. Download [mnist_train.csv](https://www.python-course.eu/data/mnist/mnist_train.csv) and [mnist_test.csv](https://www.python-course.eu/data/mnist/mnist_test.csv)
2. Place above csv files in "Learning_On_The_Go/Data/mnist"

# Code Explanation

## data.py

### shuffle()

- Takes as input a set of images and labels
- Shuffles the set of (image,label) pairs randomly


```
def shuffle(set_images, set_labels):

    shuffler = np.random.permutation(len(set_images))

    set_images_shuffled = set_images[shuffler]
    set_labels_shuffled = set_labels[shuffler]

    return set_images_shuffled, set_labels_shuffled
```

### rotate()

- Takes as input a set of images
- Rotates each image randomly by either; 0, 90, 180 or 270 degrees

```
def rotate(set_images):
    for image in set_images:
        index = random.randint(0, 3)

        np.rot90(image, k=index)

    return set_images
```

### train()

- Read 1000 lines at a time
- Split data by ","
- Extract labels and image from line
- Reshape and grey scale data
- Extract images 0-4
- Rotate and suffle images

```
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
        
```

### test()

- Read 1000 lines at a time
- Split data by ","
- Extract labels and image from line
- Reshape and grey scale data
- Extract images 5-9
- Rotate and suffle images

```
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
```

## cnn.py

### cnn()

- Create convolutional neural network
- Using Tensorflow library

```
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
```
### train()

- Take data and network as input
- Calling data.train() get 1000 (train_image,label) pairs
- Train the network with the data

```
def train(data, network):
    index = 0;
    while index != 60:
        train_images, train_labels = data.train(index)
        network.fit(x=train_images, y=train_labels, epochs=3)
        index = index + 1
```

### test()

- Take data and network as input
- Calling data.test() get 1000 unlabeled test_images
- Get predictions for each image
- Get max prediction value for each image
- If the value is lower than 0.7, then the network has correctly determined this image as unclassifible 

```
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
```

