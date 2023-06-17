import cv2
import numpy as np
import tensorflow as tf


def random_rotation(image, label=None, rotation_range=45):
    """
    Apply random rotation to the input image and label (if provided).

    Args:
        image (np.ndarray): Input image as a NumPy array.
        label (np.ndarray): Input label as a NumPy array (optional).
        rotation_range (int): Range of random rotation in degrees.

    Returns:
        augmented_image (np.ndarray): Augmented image as a NumPy array.
        augmented_label (np.ndarray): Augmented label as a NumPy array (if provided).
    """
    angle = np.random.uniform(low=-rotation_range, high=rotation_range)

    augmented_image = tf.keras.preprocessing.image.random_rotation(
        image,
        angle,
        row_axis=0,
        col_axis=1,
        channel_axis=2,
        fill_mode="nearest",
    )

    augmented_label = None
    if label is not None:
        augmented_label = tf.keras.preprocessing.image.random_rotation(
            label,
            angle,
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode="nearest",
        )

    return augmented_image, augmented_label


def random_flip(image, label=None, flip_probability=0.5):
    """
    Apply random horizontal flip to the input image and label (if provided).

    Args:
        image (np.ndarray): Input image as a NumPy array.
        label (np.ndarray): Input label as a NumPy array (optional).
        flip_probability (float): Probability of horizontal flip.

    Returns:
        augmented_image (np.ndarray): Augmented image as a NumPy array.
        augmented_label (np.ndarray): Augmented label as a NumPy array (if provided).
    """
    augmented_image = image
    augmented_label = label

    if np.random.rand() < flip_probability:
        augmented_image = tf.image.flip_left_right(image)

        if label is not None:
            augmented_label = tf.image.flip_left_right(label)

    return augmented_image, augmented_label


def random_scaling(image, label=None, scale_range=(0.8, 1.2)):
    """
    Apply random scaling to the input image and label (if provided).

    Args:
        image (np.ndarray): Input image as a NumPy array.
        label (np.ndarray): Input label as a NumPy array (optional).
        scale_range (tuple): Range of random scaling factors.

    Returns:
        augmented_image (np.ndarray): Augmented image as a NumPy array.
        augmented_label (np.ndarray): Augmented label as a NumPy array (if provided).
    """
    scale_factor = np.random.uniform(low=scale_range[0], high=scale_range[1])
    new_size = tuple((np.array(image.shape[:2]) * scale_factor).astype(int))

    augmented_image = tf.image.resize(image, new_size)

    augmented_label = None
    if label is not None:
        augmented_label = tf.image.resize(label, new_size)

    return augmented_image, augmented_label


def elastic_deformation(image, label=None, alpha=100, sigma=10):
    """
    Apply elastic deformation to the input image and label (if provided).

    Args:
        image (np.ndarray): Input image as a NumPy array.
        label (np.ndarray): Input label as a NumPy array (optional).
        alpha (float): Scaling factor for the displacement field.
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        augmented_image (np.ndarray): Augmented image as a NumPy array.
        augmented_label (np.ndarray): Augmented label as a NumPy array (if provided).
    """
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    augmented_image = tf.reshape(
        tf.gather_nd(tf.convert_to_tensor(image), indices, batch_dims=0), shape
    )

    augmented_label = None
    if label is not None:
        augmented_label = tf.reshape(
            tf.gather_nd(tf.convert_to_tensor(label), indices, batch_dims=0),
            shape,
        )

    return augmented_image, augmented_label
