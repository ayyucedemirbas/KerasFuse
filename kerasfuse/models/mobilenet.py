# Example usage
"""input_shape = (224, 224, 3)  # Input shape of the images
num_classes = 1000  # Number of output classes

model = MobileNet(input_shape, num_classes)
model.summary()"""


import tensorflow as tf
from tensorflow.keras import layers


def mobile_net(input_shape, num_classes):
    model = tf.keras.Sequential()

    # First convolution block
    model.add(
        layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="same", input_shape=input_shape
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Depthwise separable convolutions
    model.add(layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Second convolution block
    model.add(layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(128, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Depthwise separable convolutions
    model.add(layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(128, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Third convolution block
    model.add(layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Depthwise separable convolutions
    model.add(layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Fourth convolution block
    model.add(layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(512, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Depthwise separable convolutions
    for _ in range(5):
        model.add(layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2D(512, (1, 1), strides=(1, 1), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

    # Fifth convolution block
    model.add(layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(1024, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Depthwise separable convolutions
    model.add(layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(1024, (1, 1), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Global average pooling and output layer
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
