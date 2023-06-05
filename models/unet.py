import tensorflow as tf


def unet(input_shape, num_classes):
    """
    U-Net model for semantic segmentation.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): U-Net model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation="relu", padding="same")(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation="relu", padding="same")(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Decoder
    up6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding="same")(drop5)
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(conv6)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(conv7)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(conv8)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
