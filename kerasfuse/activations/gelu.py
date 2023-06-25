import tensorflow as tf


def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    # CDF stands for Cumulative Distribution Function
    return x * cdf
