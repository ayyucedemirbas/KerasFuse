import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.image import img_to_array


def grad_cam(model, image, interpolant=0.5, plot_results=True):
    """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
    using the gradients from the last convolutional layer. This function
    should work with all Keras Application listed here:
    https://keras.io/api/applications/
    Parameters:
    model (keras.model): Compiled Model with Weights Loaded
    image: Image to Perform Inference On
    plot_results (boolean): True - Function Plots using PLT
                            False - Returns Heatmap Array
    Returns:
    Heatmap Array?


    Usage:
    #load image
    test_img = cv2.imread("/content/Testing/meningioma/Te-me_0033.jpg")
    test_img = cv2.resize(test_img, (240, 240))

    #apply function
    VizGradCAM(model, img_to_array(test_img), plot_results=True)
    """
    # sanity check
    assert (
        interpolant > 0 and interpolant < 1
    ), "Heatmap Interpolation Must Be Between 0 - 1"

    # STEP 1: Preprocesss image and make prediction using our model
    # input image
    original_img = np.asarray(image, dtype=np.float32)
    # expamd dimension and get batch size
    img = np.expand_dims(original_img, axis=0)
    # predict
    prediction = model.predict(img)
    # prediction index
    prediction_idx = np.argmax(prediction)

    # STEP 2: Create new model
    # specify last convolutional layer
    last_conv_layer = next(
        x for x in model.layers[::-1] if isinstance(x, layers.Conv2D)
    )
    target_layer = model.get_layer(last_conv_layer.name)

    # compute gradient of top predicted class
    with tf.GradientTape() as tape:
        # create a model with original model inputs and the last conv_layer as the output
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        # pass the image through the base model and get the feature map
        conv2d_out, prediction = gradient_model(img)
        # prediction loss
        loss = prediction[:, prediction_idx]

    # gradient() computes the gradient using operations recorded in context of this tape
    gradients = tape.gradient(loss, conv2d_out)

    # obtain the output from shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]

    # obtain depthwise mean
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    # create a 7x7 map for aggregation
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    # multiply weight for every layer
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    # resize to image size
    activation_map = cv2.resize(
        activation_map.numpy(), (original_img.shape[1], original_img.shape[0])
    )
    # ensure no negative number
    activation_map = np.maximum(activation_map, 0)
    # convert class activation map to 0 - 255
    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min()
    )
    # rescale and convert the type to int
    activation_map = np.uint8(255 * activation_map)

    # convert to heatmap
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    # superimpose heatmap onto image
    original_img = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min())
        * 255
    )
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cvt_heatmap = img_to_array(cvt_heatmap)

    # enlarge plot
    plt.rcParams["figure.dpi"] = 100

    if plot_results is True:
        plt.imshow(
            np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant))
        )
    else:
        return cvt_heatmap
