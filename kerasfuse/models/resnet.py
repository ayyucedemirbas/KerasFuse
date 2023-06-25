import tensorflow as tf


def res_net(input_shape, num_classes):
    """
    ResNet model for image classification.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): ResNet model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocessing
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # Residual blocks
    block_sizes = [64, 128, 256, 512]
    for size in block_sizes:
        x = residual_block(x, size)

    # Global average pooling and output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def residual_block(x, filters):
    """
    Residual block for ResNet.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters.

    Returns:
        tf.Tensor: Output tensor.
    """
    shortcut = x

    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Shortcut connection
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1)(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    return x
