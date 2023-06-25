import tensorflow as tf
from tensorflow.keras import Model, layers
from transformerblock import TransformerBlock as TransformerBlock

from kerasfuse.activations import gelu


class VisionTransformer(Model):
    def __init__(
        self, num_layers, d_model, num_heads, image_size, patch_size, dropout=0.1
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_layers = [
            TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ]
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=64, activation=gelu.gelu)
        self.classifier = layers.Dense(units=10, activation="softmax")

    def extract_patches(self, images):
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, shape=(-1, self.patch_size, self.patch_size, 1))
        return patches

    def call(self, x, training):
        x = tf.image.resize(x, size=[self.image_size, self.image_size])
        x = self.extract_patches(x)
        for layer in self.encoder_layers:
            x = layer(x, training)
        x = self.layernorm(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.classifier(x)
