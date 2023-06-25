import tensorflow as tf


def dense_net(input_shape, num_classes):
    """
    DenseNet model for image classification.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): DenseNet model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocessing
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # Dense blocks
    num_layers = [6, 12, 24, 16]
    growth_rate = 32
    num_filters = 64

    for i, num_layers_block in enumerate(num_layers):
        x = dense_block(x, num_layers_block, growth_rate)

        # Transition block
        if i != len(num_layers) - 1:
            num_filters += num_layers_block * growth_rate
            x = transition_block(x, num_filters)

    # Global average pooling and output
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def dense_block(x, num_layers, growth_rate):
    """
    Dense block for DenseNet.

    Args:
        x (tf.Tensor): Input tensor.
        num_layers (int): Number of layers in the dense block.
        growth_rate (int): Growth rate (number of filters in each layer).

    Returns:
        tf.Tensor: Output tensor.
    """
    for _ in range(num_layers):
        layer = bottleneck_layer(x, growth_rate)
        x = tf.keras.layers.Concatenate()([x, layer])
    return x


def bottleneck_layer(x, growth_rate):
    """
    Bottleneck layer for DenseNet.

    Args:
        x (tf.Tensor): Input tensor.
        growth_rate (int): Growth rate (number of filters in the bottleneck layer).

    Returns:
        tf.Tensor: Output tensor.
    """
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(4 * growth_rate, 1)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(growth_rate, 3, padding="same")(x)

    return x


def transition_block(x, num_filters):
    """
    Transition block for DenseNet.

    Args:
        x (tf.Tensor): Input tensor.
        num_filters (int): Number of filters in the transition block.

    Returns:
        tf.Tensor: Output tensor.
    """
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, 1)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")(x)
    return x
