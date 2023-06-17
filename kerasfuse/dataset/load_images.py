import os

import numpy as np
import tensorflow as tf


def load_images(data_dir):
    """
    Load medical imaging data from a directory.

    Args:
        data_dir (str): Path to the directory containing the data.

    Returns:
        images (np.ndarray): Loaded images as a NumPy array.
        labels (np.ndarray): Corresponding labels as a NumPy array (if applicable).
    """
    image_paths = sorted(
        [
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith(".png")
        ]
    )
    images = []
    labels = []

    for image_path in image_paths:
        image = tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale"
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        images.append(image)

        # Assuming the labels are stored
        # in separate files with the same name as the images
        label_path = os.path.splitext(image_path)[0] + "_label.png"
        if os.path.exists(label_path):
            label = tf.keras.preprocessing.image.load_img(
                label_path, color_mode="grayscale"
            )
            label = tf.keras.preprocessing.image.img_to_array(label)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels) if labels else None

    return images, labels


def preprocess_images(images):
    """
    Preprocess medical images.

    Args:
        images (np.ndarray): Input images as a NumPy array.

    Returns:
        preprocessed_images (np.ndarray): Preprocessed images as a NumPy array.
    """
    preprocessed_images = images / 255.0  # Normalize pixel values to [0, 1]
    # Add any additional preprocessing steps here

    return preprocessed_images
