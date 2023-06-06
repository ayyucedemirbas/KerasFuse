import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for semantic segmentation.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-7).

    Returns:
        tf.Tensor: Dice coefficient.
    """
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return dice_coeff


def binary_accuracy(y_true, y_pred):
    """
    Binary accuracy for binary classification.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Binary accuracy.
    """
    y_pred = tf.round(y_pred)
    correct_predictions = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy


def categorical_accuracy(y_true, y_pred):
    """
    Categorical accuracy for multi-class classification.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Categorical accuracy.
    """
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    correct_predictions = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy


def accuracy(y_true, y_pred):
    """
    Accuracy metric for classification.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Accuracy.
    """
    correct_predictions = tf.equal(
        tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy


def precision(y_true, y_pred):
    """
    Precision metric for binary or multi-class classification.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Precision.
    """
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision


def recall(y_true, y_pred):
    """
    Recall metric for binary or multi-class classification.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Recall.
    """
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    actual_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return recall


def f1_score(y_true, y_pred):
    """
    F1 score metric for binary or multi-class classification.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1 = (2.0 * p * r) / (p + r + tf.keras.backend.epsilon())
    return f1
