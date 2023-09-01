import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# keras provides a MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalize the lights vals to range from 0-1
train_images, test_images = train_images/255.0, test_images/255.0

# create the model architecture
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)), # flatten the 28x28 image to a 1d arr
    layers.Dense(128, activation='relu'), # fully connect layers and reLU activation
    layers.Dense(10, activation='softmax') # Output layer with 10 units and softmax activation
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

model.save('trained_model.h5')