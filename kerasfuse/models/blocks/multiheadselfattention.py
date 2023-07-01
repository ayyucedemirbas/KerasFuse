import tensorflow as tf
from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.query_dense = layers.Dense(units=d_model)
        self.key_dense = layers.Dense(units=d_model)
        self.value_dense = layers.Dense(units=d_model)

        self.dense = layers.Dense(units=d_model)

    def attention(self, query, key, value):
        # Calculate dot product attention
        dot_product = tf.matmul(query, key, transpose_b=True)
        dot_product = dot_product / tf.sqrt(tf.cast(self.depth, dtype=tf.float32))
        weights = tf.nn.softmax(dot_product)

        # Calculate weighted sum of values
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # Linear layers
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Split heads
        batch_size = tf.shape(query)[0]
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Calculate dot product attention
        output, weights = self.attention(query, key, value)

        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, shape=(batch_size, -1, self.d_model))

        # Final linear layer
        output = self.dense(output)

        return output
