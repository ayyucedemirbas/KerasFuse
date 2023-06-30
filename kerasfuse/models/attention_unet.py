import tensorflow as tf


def attention_block(input1, input2):
    """
    Attention block used in the attention U-Net model.

    Args:
        input1 (Tensor): First input tensor.
        input2 (Tensor): Second input tensor.

    Returns:
        output (Tensor): Output tensor after applying attention.
    """
    g = tf.keras.layers.Conv2D(filters=input1.shape[-1], kernel_size=1)(input1)
    x = tf.keras.layers.Conv2D(filters=input2.shape[-1], kernel_size=1)(input2)
    psi = tf.keras.layers.Add()([g, x])
    psi = tf.keras.layers.Activation("relu")(psi)
    psi = tf.keras.layer.Conv2D(filters=1, kernel_size=1)(psi)
    psi = tf.keras.layers.Activation("sigmoid")(psi)
    output = tf.keras.layers.Activation("sigmoid")(psi)
    return output


def attention_unet(input_shape, num_classes):
    """
    Attention U-net model for image segmentation

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Attention U-Net model.
    """

    inputs = tf.keras.Input(shape=input_shape)

    # Encoding path
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

    # Decoding path
    up6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding="same")(drop5)
    up6 = attention_block(drop4, up6)
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(conv6)
    up7 = attention_block(conv3, up7)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(conv7)
    up8 = attention_block(conv2, up8)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(conv8)
    up9 = attention_block(conv1, up9)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
