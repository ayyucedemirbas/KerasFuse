import tensorflow as tf


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for binary classification.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
        gamma (float): Focusing parameter (default: 2.0).
        alpha (float): Class balancing parameter (default: 0.25).

    Returns:
        tf.Tensor: Focal loss.
    """
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = -alpha * y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred) - (
        1.0 - alpha
    ) * (1.0 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
    return tf.reduce_mean(loss)
