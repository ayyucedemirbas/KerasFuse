from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import layers


class AutoEncoder(tf.keras.Model):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        inter_channels: list | None = None,
        inter_dilations: list | None = None,
        num_inter_units: int = 2,
        act: str | None = "prelu",
        norm: str | None = "instance",
        dropout: float | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError(
                "Autoencoder expects matching number of channels and strides"
            )

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        self.encode, self.encoded_channels = self._get_encode_module(
            self.encoded_channels, channels, strides
        )
        self.intermediate, self.encoded_channels = self._get_intermediate_module(
            self.encoded_channels, num_inter_units
        )
        self.decode, _ = self._get_decode_module(
            self.encoded_channels, decode_channel_list, strides[::-1] or [1]
        )

    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> tuple[layers.Layer, int]:
        encode = tf.keras.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add(layer)
            layer_channels = c

        return encode, layer_channels

    def _get_intermediate_module(
        self, in_channels: int, num_inter_units: int
    ) -> tuple[layers.Layer, int]:
        intermediate = tf.keras.layers.Identity()
        layer_channels = in_channels

        if self.inter_channels:
            intermediate = tf.keras.Sequential()

            for i, (dc, di) in enumerate(
                zip(self.inter_channels, self.inter_dilations)
            ):
                if self.num_inter_units > 0:
                    unit = ResidualUnit(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        subunits=self.num_inter_units,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                    )
                else:
                    unit = Convolution(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                    )

                intermediate.add(unit)
                layer_channels = dc

        return intermediate, layer_channels

    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> tuple[layers.Layer, int]:
        decode = tf.keras.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(
                layer_channels, c, s, i == (len(strides) - 1)
            )
            decode.add(layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> layers.Layer:
        if self.num_res_units > 0:
            mod = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
            return mod

        mod = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )
        return mod

    def _get_decode_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> layers.Layer:
        decode = tf.keras.Sequential()

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=True,
        )

        decode.add(conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )

            decode.add(ru)

        return decode

    def call(self, x):
        x = self.encode(x)
        x = self.intermediate(x)
        x = self.decode(x)
        return x
