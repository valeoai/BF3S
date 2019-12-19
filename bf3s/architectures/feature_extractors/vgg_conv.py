import torchvision.models as models

import bf3s.architectures.feature_extractors.utils as utils
import bf3s.architectures.tools as tools


class VGGConv(utils.SequentialFeatureExtractorAbstractClass):
    def __init__(self, arch, pool="avg"):
        assert (
            arch == "vgg11"
            or arch == "vgg11_bn"
            or arch == "vgg13"
            or arch == "vgg13_bn"
            or arch == "vgg16"
            or arch == "vgg16_bn"
            or arch == "vgg19"
            or arch == "vgg19_bn"
        )

        net = models.__dict__[arch](num_classes=10)

        all_feat_names = []
        feature_blocks = []
        features = net.features[:-1]
        feature_blocks.append(features)
        all_feat_names.append("features")

        assert pool == "none" or pool == "avg" or pool == "max"
        if pool == "max" or pool == "avg":
            feature_blocks.append(tools.GlobalPooling(pool_type=pool))
            all_feat_names.append("GlobalPooling")

        super().__init__(all_feat_names, feature_blocks)


def create_model(opt):
    return VGGConv(arch=opt["arch"], pool=opt["pool"])
