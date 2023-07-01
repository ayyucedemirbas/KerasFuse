import unittest

import tensorflow as tf

from kerasfuse.metrics import (
    accuracy,
    balanced_accuracy,
    binary_accuracy,
    categorical_accuracy,
    dice_coefficient,
    f1_score,
    precision,
    recall,
    specificity,
)


class TestMetrics(unittest.TestCase):
    """
    Unittest for metrics module.
    """

    def test_dice_coefficient(self):
        y_true = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 0], [1, 0]], dtype=tf.float32)
        result = dice_coefficient(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.5, places=6)

    def test_binary_accuracy(self):
        y_true = tf.constant([1, 0, 1, 0], dtype=tf.float32)
        y_pred = tf.constant([1, 0, 0, 1], dtype=tf.float32)
        result = binary_accuracy(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.5, places=6)

    def test_categorical_accuracy(self):
        y_true = tf.constant([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=tf.float32)
        result = categorical_accuracy(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.0, places=6)

    def test_accuracy(self):
        y_true = tf.constant([[1, 0], [0, 1], [1, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 0], [0, 1], [0, 1]], dtype=tf.float32)
        result = accuracy(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.6666667, places=6)

    def test_precision(self):
        y_true = tf.constant([1, 0, 1, 0], dtype=tf.float32)
        y_pred = tf.constant([1, 1, 0, 0], dtype=tf.float32)
        result = precision(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.5, places=6)

    def test_recall(self):
        y_true = tf.constant([1, 0, 1, 0], dtype=tf.float32)
        y_pred = tf.constant([1, 1, 0, 0], dtype=tf.float32)
        result = recall(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.5, places=6)

    def test_f1_score(self):
        y_true = tf.constant([1, 0, 1, 0], dtype=tf.float32)
        y_pred = tf.constant([1, 1, 0, 0], dtype=tf.float32)
        result = f1_score(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.49999994, places=6)

    def test_specificity(self):
        y_true = tf.constant([1, 0, 1, 0], dtype=tf.float32)
        y_pred = tf.constant([1, 1, 0, 0], dtype=tf.float32)
        result = specificity(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.5, places=6)

    def test_balanced_accuracy(self):
        y_true = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=tf.float32)
        result = balanced_accuracy(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 0.5, places=6)
