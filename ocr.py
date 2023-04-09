"""
OCR DeepConv
Copyright (c) 2023 Alexander Leong
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2                          # for colour conversion
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import tensorflow as tf             # for bulk image resize
from scipy.io import loadmat        # to load mat files
from tensorflow import keras
from tensorflow.keras import layers

def load_data(train_path, test_path):
    # load files
    train = loadmat(train_path)
    test = loadmat(test_path)

    # transpose, such that dimensions are (sample, width, height, channels), and divide by 255.0
    train_X = np.transpose(train['train_X'], (3, 0, 1, 2)) / 255.0
    train_Y = train['train_Y']
    # change labels '10' to '0' for compatability with keras/tf. The label '10' denotes the digit '0'
    train_Y[train_Y == 10] = 0
    train_Y = np.reshape(train_Y, -1)

    # transpose, such that dimensions are (sample, width, height, channels), and divide by 255.0
    test_X = np.transpose(test['test_X'], (3, 0, 1, 2)) / 255.0
    test_Y = test['test_Y']
    # change labels '10' to '0' for compatability with keras/tf. The label '10' denotes the digit '0'
    test_Y[test_Y == 10] = 0
    test_Y = np.reshape(test_Y, -1)

    # return loaded data
    return train_X, train_Y, test_X, test_Y

def plot_images(x, y):
    fig = plt.figure(figsize=[15, 180])
    for i in range(1000):
        ax = fig.add_subplot(100, 10, i + 1)
        ax.imshow(x[i,:], cmap="gray", vmin=0, vmax=1)
        ax.set_title(y[i])
        ax.axis('off')

def resize(images, new_size):
    # tensorflow has an image resize funtion that can do this in bulk
    # note the conversion back to numpy after the resize
    return tf.image.resize(images, new_size).numpy()
          
def convert_to_grayscale(images):
    # storage for converted images
    gray = []
    # loop through images
    for i in range(len(images)):
        # convert each image using openCV
        grayImage = cv2.cvtColor(images[i,:], cv2.COLOR_BGR2GRAY)
        gray.append(grayImage)
    # pack converted list as an array and return
    return np.expand_dims(np.array(gray), axis = -1) # np.array(gray)

train_X, train_Y, test_X, test_Y = load_data('./train.mat', './test.mat')

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 1)

train_X = convert_to_grayscale(resize(train_X, (32, 32)))
test_X = convert_to_grayscale(resize(test_X, (32, 32)))

# Scale images to the [0, 1] range
x_train = train_X.astype("float32") / 255
x_test = test_X.astype("float32") / 255


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(train_Y, num_classes)
y_test = keras.utils.to_categorical(test_Y, num_classes)

class ColorInversionLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ColorInversionLayer, self).__init__()

    def call(self, inputs):
      p = 0.5
      if  tf.random.uniform([]) < p:
        inputs = (1-inputs)
      return inputs

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.ZeroPadding2D(padding=(8, 8)),
        layers.BatchNormalization(),
        ColorInversionLayer(),
        layers.RandomRotation(0.05),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 50
epochs = 80

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# plot_images(train_X, train_Y)

model.fit(train_X, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(train_X, y_train, verbose=0)
print("Train loss:", score[0])
print("Train accuracy:", score[1])
score = model.evaluate(test_X, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
