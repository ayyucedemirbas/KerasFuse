import tensorflow as tf


def train_model(model, train_dataset, val_dataset, loss_fn, optimizer, metrics, epochs):
    """
    Train the model using the given datasets and parameters.

    Args:
        model (tf.keras.Model): The model to train.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        loss_fn: Loss function for model optimization.
        optimizer: The optimizer to use for training.
        metrics (list): List of evaluation metrics.
        epochs (int): Number of training epochs.
    """
    train_loss = tf.keras.metrics.Mean()
    train_metrics = [tf.keras.metrics.MeanMetricWrapper(metric) for metric in metrics]
    val_loss = tf.keras.metrics.Mean()
    val_metrics = [tf.keras.metrics.MeanMetricWrapper(metric) for metric in metrics]

    # Define training and validation steps
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        for metric in train_metrics:
            metric(labels, predictions)

    @tf.function
    def val_step(inputs, labels):
        predictions = model(inputs, training=False)
        loss = loss_fn(labels, predictions)
        val_loss(loss)
        for metric in val_metrics:
            metric(labels, predictions)

    # Training loop
    for epoch in range(epochs):
        # Reset metrics
        train_loss.reset_states()
        val_loss.reset_states()
        for metric in train_metrics + val_metrics:
            metric.reset_states()

        # Training
        for inputs, labels in train_dataset:
            train_step(inputs, labels)

        # Validation
        for val_inputs, val_labels in val_dataset:
            val_step(val_inputs, val_labels)

        # Print epoch results
        template = "Epoch {}, Train Loss: {}, Val Loss: {}"
        metric_template = ", Train {}: {:.4f}, Val {}: {:.4f}"
        metrics_string = ""
        for metric, train_metric, val_metric in zip(
            metrics, train_metrics, val_metrics
        ):
            metrics_string += metric_template.format(
                metric.__name__,
                train_metric.result(),
                metric.__name__,
                val_metric.result(),
            )
        print(
            template.format(epoch + 1, train_loss.result(), val_loss.result())
            + metrics_string
        )
