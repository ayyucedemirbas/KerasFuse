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
    loss = -alpha * y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred) - \
           (1.0 - alpha) * (1.0 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
    return tf.reduce_mean(loss)


def binary_cross_entropy(y_true, y_pred):
    """
    Binary cross-entropy loss for binary classification.
    
    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
        
    Returns:
        tf.Tensor: Binary cross-entropy loss.
    """
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(loss)


def categorical_cross_entropy(y_true, y_pred):
    """
    Categorical cross-entropy loss for multi-class classification.
    
    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
        
    Returns:
        tf.Tensor: Categorical cross-entropy loss.
    """
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return tf.reduce_mean(loss)
