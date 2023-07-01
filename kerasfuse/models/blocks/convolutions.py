import tensorflow as tf
from collections.abc import Sequence
from kerasfuse.models.layers.convutils import same_padding,stride_minus_kernel_padding


class Convolution(tf.keras.Sequential):
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = tf.keras.layers.Conv2DTranspose if is_transposed else tf.keras.layers.Conv2D


        conv: tf.keras.layers.Layer
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                use_bias=bias,
                dilation_rate=dilation,
            )
        else:
            conv = conv_type(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=bias,
            )

        self.add(conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add(
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )