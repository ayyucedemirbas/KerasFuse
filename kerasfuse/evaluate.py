import tensorflow as tf


def evaluate_model(model, test_dataset, loss_fn, metrics):
    """
    Evaluate the model using the given test dataset and metrics.

    Args:
        model (tf.keras.Model): The model to evaluate.
        test_dataset (tf.data.Dataset): Test dataset.
        loss_fn: Loss function for model evaluation.
        metrics (list): List of evaluation metrics.
    """
    test_loss = tf.keras.metrics.Mean()
    test_metrics = [tf.keras.metrics.MeanMetricWrapper(metric) for metric in metrics]

    @tf.function
    def test_step(inputs, labels):
        predictions = model(inputs, training=False)
        loss = loss_fn(labels, predictions)
        test_loss(loss)
        for metric in test_metrics:
            metric(labels, predictions)

    # Evaluation loop
    for inputs, labels in test_dataset:
        test_step(inputs, labels)

    # Print results
    template = "Test Loss: {}"
    metric_template = ", Test {}: {:.4f}"
    metrics_string = ""
    for metric, test_metric in zip(metrics, test_metrics):
        metrics_string += metric_template.format(metric.__name__, test_metric.result())
    print(template.format(test_loss.result()) + metrics_string)
