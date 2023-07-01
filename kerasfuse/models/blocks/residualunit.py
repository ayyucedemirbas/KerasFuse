from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from kerasfuse.models.layers.convutils import same_padding

from .convolutions import Convolution


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = tf.keras.Sequential()
        self.residual = tf.identity
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )

            self.conv.add(unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if (
                np.prod(strides) == 1
            ):  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = tf.keras.layers.Conv[
                tf.keras.layers.Conv.CONV, self.spatial_dims
            ]
            self.residual = conv_type(
                in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        res: tf.Tensor = self.residual(x)  # create the additive residual from x
        cx: tf.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
