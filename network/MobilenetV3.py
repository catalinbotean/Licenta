from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor

from network.mynn import forgiving_state_restore
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .instance_whitening import InstanceWhitening

__all__ = ["MobileNetV3", "mobilenet_v3"]

model_urls = {
    'mobilenet_v3': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
}


class SElayer(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU(),
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU(),
        iw: int = 0,
    ) -> None:

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.iw = iw

        if iw == 1:
            instance_norm_layer = InstanceWhitening(out_planes)
        elif iw == 2:
            instance_norm_layer = InstanceWhitening(out_planes)
        elif iw == 3:
            instance_norm_layer = nn.InstanceNorm2d(out_planes, affine=False)
        elif iw == 4:
            instance_norm_layer = nn.InstanceNorm2d(out_planes, affine=True)
        else:
            instance_norm_layer = nn.Sequential()

        super(ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
                norm_layer(out_planes),
                activation_layer,
                instance_norm_layer
            )


    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error in BN forward path")
            return

        for i, module in enumerate(self):
            if i == len(self) - 1:
                if self.iw >= 1:
                    if self.iw == 1 or self.iw == 2:
                        x, w = self.instance_norm_layer(x)
                        w_arr.append(w)
                    else:
                        x = self.instance_norm_layer(x)
            else:
                x = module(x)

        return [x, w_arr]


Conv2dNormActivation = ConvBNReLU


class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
        iw: int = 0,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.iw = iw

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.iw = cnf.iw
        self.expand_ratio = cnf.expanded_channels
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish() if cnf.use_hs else nn.ReLU()

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    iw=cnf.iw
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                iw=cnf.iw
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None, iw=cnf.iw
            )
        )
        if cnf.iw == 1:
            self.instance_norm_layer = InstanceWhitening(cnf.out_channels)
        elif cnf.iw == 2:
            self.instance_norm_layer = InstanceWhitening(cnf.out_channels)
        elif cnf.iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(cnf.out_channels, affine=False)
        elif cnf.iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(cnf.out_channels, affine=False)
        else:
            self.instance_norm_layer = nn.Sequential()
        self.conv = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            x = x_tuple[0]
        else:
            print("error in invert residual forward path")
            return
        if self.expand_ratio != 1:
            x_tuple = self.conv[0](x_tuple)
            x_tuple = self.conv[1](x_tuple)
            conv_x = x_tuple[0]
            w_arr = x_tuple[1]
            conv_x = self.conv[2](conv_x)
            conv_x = self.conv[3](conv_x)
        else:
            x_tuple = self.conv[0](x_tuple)
            conv_x = x_tuple[0]
            w_arr = x_tuple[1]
            conv_x = self.conv[1](conv_x)
            conv_x = self.conv[2](conv_x)

        if self.use_res_connect:
            x = x + conv_x
        else:
            x = conv_x

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                x, w = self.instance_norm_layer(x)
                w_arr.append(w)
            else:
                x = self.instance_norm_layer(x)

        return [x, w_arr]


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        iw: list = [0, 0, 0, 0, 0, 0, 0],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish(),
            )
        )
        feature_count = 0
        iw_layer = [0, 2, 7, 11, 12]
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            feature_count += 1
            if feature_count in iw_layer:
                layer = iw_layer.index(feature_count)
                cnf.iw = iw[layer+2]
                layers.append(block(cnf, norm_layer))
            else:
                cnf.iw = 0
                layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish(),
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, iw: int = 0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult, iw=iw)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def mobilenet_v3(pretrained: bool = False, progress: bool = True, **kwargs: Any,) -> MobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v3'],
                                              progress=progress)
        forgiving_state_restore(model, state_dict)
    return model
