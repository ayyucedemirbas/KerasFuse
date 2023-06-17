import tensorflow as tf
from keras.layers.normalization.layer_normalization import LayerNormalization
from tensorflow import keras
from tensorflow.python.keras.layers import Activation, Dense, Dropout, Layer

"""
Author: Khalid Salama
https://keras.io/examples/vision/mlp_image_classification/
"""


class Patches(Layer):
    def __init__(self, patch_size, num_patches):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches

    class MLPMixerLayer(Layer):
        def __init__(
            self,
            num_patches,
            hidden_units,
            dropout_rate,
            embedding_dim,
            *args,
            **kwargs
        ):
            super().__init__(*args, **kwargs)

            self.mlp1 = keras.Sequential(
                [
                    Dense(units=num_patches),
                    Activation(keras.activations.gelu),
                    Dense(units=num_patches),
                    Dropout(rate=dropout_rate),
                ]
            )
            self.mlp2 = keras.Sequential(
                [
                    Dense(units=num_patches),
                    Activation(keras.activations.gelu),
                    Dense(units=embedding_dim),
                    Dropout(rate=dropout_rate),
                ]
            )
            self.normalize = LayerNormalization(epsilon=1e-6)

        def call(self, inputs):
            # Apply layer normalization.
            x = self.normalize(inputs)
            # Transpose inputs from [num_batches, num_patches,
            # hidden_units] to [num_batches, hidden_units, num_patches].
            x_channels = tf.linalg.matrix_transpose(x)
            # Apply mlp1 on each channel independently.
            mlp1_outputs = self.mlp1(x_channels)
            # Transpose mlp1_outputs from [num_batches,
            # hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
            mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
            # Add skip connection.
            x = mlp1_outputs + inputs
            # Apply layer normalization.
            x_patches = self.normalize(x)
            # Apply mlp2 on each patch independtenly.
            mlp2_outputs = self.mlp2(x_patches)
            # Add skip connection.
            x = x + mlp2_outputs
            return x
