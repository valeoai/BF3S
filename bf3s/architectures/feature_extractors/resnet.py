import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet_utils

import bf3s.architectures.feature_extractors.utils as utils
import bf3s.architectures.tools as tools


class ResNet(utils.SequentialFeatureExtractorAbstractClass):
    def __init__(self, arch, pool="avg"):
        assert (
            arch == "resnet10"
            or arch == "resnet18"
            or arch == "resnet34"
            or arch == "resnet50"
            or arch == "resnet101"
            or arch == "resnet152"
        )

        if arch == "resnet10":
            net = resnet_utils.ResNet(
                block=resnet_utils.BasicBlock, layers=[1, 1, 1, 1], num_classes=10
            )
        else:
            net = models.__dict__[arch](num_classes=10)

        all_feat_names = []
        feature_blocks = []

        # 1st conv before any network block
        conv1 = nn.Sequential()
        conv1.add_module("Conv", net.conv1)
        conv1.add_module("bn", net.bn1)
        conv1.add_module("relu", net.relu)
        conv1.add_module("maxpool", net.maxpool)
        feature_blocks.append(conv1)
        all_feat_names.append("conv1")

        # 1st block.
        feature_blocks.append(net.layer1)
        all_feat_names.append("block1")

        # 2nd block.
        feature_blocks.append(net.layer2)
        all_feat_names.append("block2")

        # 3rd block.
        feature_blocks.append(net.layer3)
        all_feat_names.append("block3")

        # 4th block.
        feature_blocks.append(net.layer4)
        all_feat_names.append("block4")

        assert pool == "none" or pool == "avg" or pool == "max"
        if pool == "max" or pool == "avg":
            feature_blocks.append(tools.GlobalPooling(pool_type=pool))
            all_feat_names.append("GlobalPooling")

        super().__init__(all_feat_names, feature_blocks)


def create_model(opt):
    return ResNet(arch=opt["arch"], pool=opt["pool"])
