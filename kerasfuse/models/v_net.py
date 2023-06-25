""" Usage example
input_shape = (128, 128, 128, 1)  # Adjust input shape according to your data
model = VNet(input_shape)
model.summary()"""


from tensorflow.keras.layers import (
    Concatenate,
    Conv3D,
    Conv3DTranspose,
    Input,
    MaxPooling3D,
)
from tensorflow.keras.models import Model


def vnet(input_shape):
    inputs = Input(input_shape)

    # Encoding path
    conv1 = Conv3D(16, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv3D(16, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv3D(32, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv3D(64, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv3D(128, 3, activation="relu", padding="same")(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Bottleneck
    conv5 = Conv3D(256, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv3D(256, 3, activation="relu", padding="same")(conv5)

    # Decoding path
    up6 = Conv3DTranspose(128, 2, strides=(2, 2, 2), padding="same")(conv5)
    up6 = Concatenate()([up6, conv4])
    conv6 = Conv3D(128, 3, activation="relu", padding="same")(up6)
    conv6 = Conv3D(128, 3, activation="relu", padding="same")(conv6)

    up7 = Conv3DTranspose(64, 2, strides=(2, 2, 2), padding="same")(conv6)
    up7 = Concatenate()([up7, conv3])
    conv7 = Conv3D(64, 3, activation="relu", padding="same")(up7)
    conv7 = Conv3D(64, 3, activation="relu", padding="same")(conv7)

    up8 = Conv3DTranspose(32, 2, strides=(2, 2, 2), padding="same")(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = Conv3D(32, 3, activation="relu", padding="same")(up8)
    conv8 = Conv3D(32, 3, activation="relu", padding="same")(conv8)

    up9 = Conv3DTranspose(16, 2, strides=(2, 2, 2), padding="same")(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = Conv3D(16, 3, activation="relu", padding="same")(up9)
    conv9 = Conv3D(16, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = Conv3D(1, 1, activation="sigmoid")(conv9)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model
