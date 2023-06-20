import tensorflow as tf


def dice_loss(y_true, y_pred):
    """
    Dice loss for semantic segmentation.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Dice loss.
    """
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score
    return dice_loss
