import math

import torch.nn as nn

import bf3s.architectures.feature_extractors.utils as utils
import bf3s.architectures.tools as tools


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, pre_relu=True, post_relu=False, pool=True):
        super().__init__()
        self.layers = nn.Sequential()

        if pre_relu:
            self.layers.add_module("PreReLU", nn.ReLU(inplace=False))

        self.layers.add_module(
            "Conv",
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))

        if pool:
            self.layers.add_module("MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        if post_relu:
            self.layers.add_module("PostReLU", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.layers(x)
        return out


class ConvNetPreAct(utils.SequentialFeatureExtractorAbstractClass):
    def __init__(self, opt):
        self.in_planes = opt["in_planes"]
        self.out_planes = opt["out_planes"]
        self.num_stages = opt["num_stages"]
        self.average_end = opt["average_end"] if ("average_end" in opt) else False

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert type(self.out_planes) == list
        assert len(self.out_planes) == self.num_stages

        num_planes = [self.in_planes,] + self.out_planes
        use_post_relu = opt["userelu"] if ("userelu" in opt) else True
        use_pre_relu = opt["use_pre_relu"] if ("use_pre_relu" in opt) else False

        self.use_pool = opt["use_pool"] if ("use_pool" in opt) else None
        if self.use_pool is None:
            self.use_pool = [True for i in range(self.num_stages)]
        assert len(self.use_pool) == self.num_stages

        feature_blocks = []
        for i in range(self.num_stages):
            feature_blocks.append(
                ConvBlock(
                    num_planes[i],
                    num_planes[i + 1],
                    pre_relu=(use_pre_relu if (i == 0) else True),
                    post_relu=(use_post_relu if i == (self.num_stages - 1) else False),
                    pool=self.use_pool[i],
                )
            )

        all_feat_names = ["conv" + str(s + 1) for s in range(self.num_stages)]

        if self.average_end:
            feature_blocks.append(tools.GlobalPooling(pool_type="avg"))
            all_feat_names.append("GlobalAvgPooling")

        super().__init__(all_feat_names, feature_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def create_model(opt):
    return ConvNetPreAct(opt)
