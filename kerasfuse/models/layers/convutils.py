import numpy as np
import tensorflow as tf


def same_padding(kernel_size, dilation=1):
    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            "Same padding not available for"
            + f"kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def stride_minus_kernel_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def calculate_out_shape(in_shape, kernel_size, stride, padding):
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_shape_np = (
        (in_shape_np - kernel_size_np + padding_np + padding_np) // stride_np
    ) + 1
    out_shape = tuple(int(s) for s in out_shape_np)

    return out_shape


def gaussian_1d(sigma, truncated=4.0, approx="erf", normalize=False):
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    if truncated <= 0.0:
        raise ValueError(f"truncated must be positive, got {truncated}.")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = tf.range(-tail, tail + 1, dtype=tf.float32)
        t = 0.70710678 / tf.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = tf.clip_by_value(out, clip_value_min=0)
    elif approx.lower() == "sampled":
        x = tf.range(-tail, tail + 1, dtype=tf.float32)
        out = tf.exp(-0.5 / (sigma * sigma) * x**2)
        if not normalize:
            out = out / (2.5066282 * sigma)
    elif approx.lower() == "scalespace":
        sigma2 = sigma * sigma
        out_pos = [None] * (tail + 1)
        out_pos[0] = _modified_bessel_0(sigma2)
        out_pos[1] = _modified_bessel_1(sigma2)
        for k in range(2, len(out_pos)):
            out_pos[k] = _modified_bessel_i(k, sigma2)
        out = out_pos[:0:-1]
        out.extend(out_pos)
        out = tf.stack(out) * tf.exp(-sigma2)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / tf.reduce_sum(out) if normalize else out


def polyval(coef, x):
    device = x.device if isinstance(x, tf.Tensor) else None
    coef = tf.convert_to_tensor(coef, dtype=tf.float32, device=device)
    if coef.ndim == 0 or (len(coef) < 1):
        return tf.zeros(x.shape)
    x = tf.convert_to_tensor(x, dtype=tf.float32, device=device)
    ans = coef[0]
    for c in coef[1:]:
        ans = ans * x + c
    return ans


def _modified_bessel_0(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if tf.abs(x) < 3.75:
        y = x * x / 14.0625
        return polyval(
            [0.45813e-2, 0.360768e-1, 0.2659732, 1.2067492, 3.0899424, 3.5156229, 1.0],
            y,
        )
    ax = tf.abs(x)
    y = 3.75 / ax
    _coef = [
        0.392377e-2,
        -0.1647633e-1,
        0.2635537e-1,
        -0.2057706e-1,
        0.916281e-2,
        -0.157565e-2,
        0.225319e-2,
        0.1328592e-1,
        0.39894228,
    ]
    return polyval(_coef, y) * tf.exp(-ax) / tf.sqrt(ax)


def _modified_bessel_1(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if tf.abs(x) < 3.75:
        y = x * x / 14.0625
        _coef = [
            0.32411e-3,
            0.301532e-2,
            0.2658733e-1,
            0.15084934,
            0.51498869,
            0.87890594,
            0.5,
        ]
        return tf.abs(x) * polyval(_coef, y)
    ax = tf.abs(x)
    y = 3.75 / ax
    _coef = [
        -0.420059e-2,
        0.1787654e-1,
        -0.2895312e-1,
        0.2282967e-1,
        -0.1031555e-1,
        0.163801e-2,
        -0.362018e-2,
        -0.3988024e-1,
        0.39894228,
    ]
    ans = polyval(_coef, y) * tf.exp(ax) / tf.sqrt(ax)
    return -ans if x < 0.0 else ans


def _modified_bessel_i(n, x):
    if n < 2:
        raise ValueError(f"n must be greater than 1, got n={n}.")
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if x == 0.0:
        return x
    device = x.device
    tox = 2.0 / tf.abs(x)
    ans, bip, bi = (
        tf.constant(0.0, device=device),
        tf.constant(0.0, device=device),
        tf.constant(1.0, device=device),
    )
    m = int(2 * (n + np.floor(np.sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        if abs(bi) > 1.0e10:
            ans = ans * 1.0e-10
            bi = bi * 1.0e-10
            bip = bip * 1.0e-10
        if j == n:
            ans = bip
    ans = ans * _modified_bessel_0(x) / bi
    return -ans if x < 0.0 and (n % 2) == 1 else ans


def same_padding(kernel_size, dilation=1):
    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            "Same padding not available for "
            + f"kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def stride_minus_kernel_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def calculate_out_shape(in_shape, kernel_size, stride, padding):
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_shape_np = (
        (in_shape_np - kernel_size_np + padding_np + padding_np) // stride_np
    ) + 1
    out_shape = tuple(int(s) for s in out_shape_np)

    return out_shape


def gaussian_1d(sigma, truncated=4.0, approx="erf", normalize=False):
    sigma = tf.convert_to_tensor(
        sigma,
        dtype=tf.float32,
        device=sigma.device if isinstance(sigma, tf.Tensor) else None,
    )
    device = sigma.device
    if truncated <= 0.0:
        raise ValueError(f"truncated must be positive, got {truncated}.")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = tf.range(-tail, tail + 1, dtype=tf.float32, device=device)
        t = 0.70710678 / tf.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = tf.clip_by_value(out, clip_value_min=0)
    elif approx.lower() == "sampled":
        x = tf.range(-tail, tail + 1, dtype=tf.float32, device=sigma.device)
        out = tf.exp(-0.5 / (sigma * sigma) * x**2)
        if not normalize:  # compute the normalizer
            out = out / (2.5066282 * sigma)
    elif approx.lower() == "scalespace":
        sigma2 = sigma * sigma
        out_pos = [None] * (tail + 1)
        out_pos[0] = _modified_bessel_0(sigma2)
        out_pos[1] = _modified_bessel_1(sigma2)
        for k in range(2, len(out_pos)):
            out_pos[k] = _modified_bessel_i(k, sigma2)
        out = out_pos[:0:-1]
        out.extend(out_pos)
        out = tf.stack(out) * tf.exp(-sigma2)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / tf.reduce_sum(out) if normalize else out


def separable_conv1d(
    x, depthwise_filter, pointwise_filter, stride=1, padding="same", dilation=1
):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    depthwise_filter = tf.convert_to_tensor(depthwise_filter, dtype=tf.float32)
    pointwise_filter = tf.convert_to_tensor(pointwise_filter, dtype=tf.float32)

    stride_np = np.atleast_1d(stride)
    dilation_np = np.atleast_1d(dilation)

    if x.ndim != 3:
        raise ValueError(f"Input 'x' must have 3 dimensions, got {x.ndim} dimensions.")
    if depthwise_filter.ndim != 2:
        raise ValueError(
            f"Depthwise filter must have 2 dimensions, "
            f"got {depthwise_filter.ndim} dimensions."
        )
    if pointwise_filter.ndim != 2:
        raise ValueError(
            f"Pointwise filter must have 2 dimensions, "
            f"got {pointwise_filter.ndim} dimensions."
        )
    if depthwise_filter.shape[1] != x.shape[1]:
        raise ValueError(
            f"Depthwise filter shape[1] ({depthwise_filter.shape[1]}) "
            f"must match input shape[1] ({x.shape[1]})."
        )
    if pointwise_filter.shape[1] != depthwise_filter.shape[0]:
        raise ValueError(
            f"Pointwise filter shape[1] ({pointwise_filter.shape[1]}) "
            f"must match depthwise filter shape[0] ({depthwise_filter.shape[0]})."
        )

    padding_np = np.atleast_1d(padding)
    input_shape_np = np.array(x.shape)

    if padding == "same":
        padding_np = same_padding(
            depthwise_filter.shape[0], dilation_np
        )  # depthwise filter size is the kernel size
        out_shape_np = input_shape_np
        out_shape_np[-1] = calculate_out_shape(
            in_shape=input_shape_np[-1],
            kernel_size=depthwise_filter.shape[0],
            stride=stride_np,
            padding=padding_np,
        )
    elif padding == "valid":
        padding_np = (0, 0)
        out_shape_np = input_shape_np
        out_shape_np[-1] = calculate_out_shape(
            in_shape=input_shape_np[-1],
            kernel_size=depthwise_filter.shape[0],
            stride=stride_np,
            padding=padding_np,
        )
    else:
        raise ValueError(f"Invalid padding: {padding}")

    depthwise_filter_expanded = tf.expand_dims(depthwise_filter, axis=0)
    depthwise_filter_expanded = tf.expand_dims(depthwise_filter_expanded, axis=-1)
    pointwise_filter_expanded = tf.expand_dims(pointwise_filter, axis=0)
    pointwise_filter_expanded = tf.expand_dims(pointwise_filter_expanded, axis=-1)

    x_padded = tf.pad(x, [[0, 0], [padding_np[0], padding_np[0]], [0, 0]])
    x_reshaped = tf.image.extract_patches(
        x_padded,
        sizes=[1, depthwise_filter.shape[0], 1, 1],
        strides=[1, stride_np, 1, 1],
        rates=[1, dilation_np, 1, 1],
        padding="VALID",
    )
    x_reshaped = tf.reshape(
        x_reshaped, [x.shape[0], out_shape_np[1], depthwise_filter.shape[0], x.shape[2]]
    )

    out = tf.einsum("bijc,bcjk->bijk", x_reshaped, depthwise_filter_expanded)
    out = tf.einsum("bijk,bck->bijc", out, pointwise_filter_expanded)
    out_shape_final = tf.concat(
        [out_shape_np[:2], tf.constant([out_shape_np[-1]])], axis=0
    )
    out = tf.reshape(out, out_shape_final)

    return out
