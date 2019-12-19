import math

import torch.nn as nn

import bf3s.architectures.feature_extractors.wide_resnet as wrn_utils


class NetworkBlockV2(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, kernel_sizes=3
    ):
        super().__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes,] * nb_layers

        layers = []
        for i in range(nb_layers):
            in_planes_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(
                block(
                    in_planes_arg,
                    out_planes,
                    stride_arg,
                    drop_rate,
                    kernel_size=kernel_sizes[i],
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResnetBlock(nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        num_layers,
        stride=2,
        drop_rate=0.0,
        kernel_sizes=None,
    ):
        super().__init__()

        self.block = nn.Sequential()
        if kernel_sizes is None:
            self.block.add_module(
                "Block",
                wrn_utils.NetworkBlock(
                    nb_layers=num_layers,
                    in_planes=num_channels_in,
                    out_planes=num_channels_out,
                    block=wrn_utils.BasicBlock,
                    stride=stride,
                    drop_rate=drop_rate,
                ),
            )
        else:
            self.block.add_module(
                "Block",
                NetworkBlockV2(
                    nb_layers=num_layers,
                    in_planes=num_channels_in,
                    out_planes=num_channels_out,
                    block=wrn_utils.BasicBlock,
                    stride=stride,
                    drop_rate=drop_rate,
                    kernel_sizes=kernel_sizes,
                ),
            )

        self.block.add_module("BN", nn.BatchNorm2d(num_channels_out))
        self.block.add_module("ReLU", nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


def create_model(opt):
    return WideResnetBlock(
        num_channels_in=opt["num_channels_in"],
        num_channels_out=opt["num_channels_out"],
        num_layers=opt["num_layers"],
        stride=opt["stride"],
        drop_rate=(opt["drop_rate"] if ("drop_rate" in opt) else 0.0),
        kernel_sizes=(opt["kernel_sizes"] if ("kernel_sizes" in opt) else None),
    )
