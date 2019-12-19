from torch import nn

from bf3s.architectures.classifiers import classifier
from bf3s.architectures.feature_extractors import convnet
from bf3s.architectures.feature_extractors import convnet_pre_act
from bf3s.architectures.feature_extractors import resnet_block
from bf3s.architectures.feature_extractors import wrn_block


class ConvnetPlusClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        convnet_type = opt["convnet_type"]

        if convnet_type == "convnet":
            convolutional_net = convnet.create_model(opt["convnet_opt"])
        elif convnet_type == "convnet_pre_act":
            convolutional_net = convnet_pre_act.create_model(opt["convnet_opt"])
        elif convnet_type == "resnet_block":
            convolutional_net = resnet_block.create_model(opt["convnet_opt"])
        elif convnet_type == "wrn_block":
            convolutional_net = wrn_block.create_model(opt["convnet_opt"])
        else:
            raise KeyError(convnet_type)

        self.layers = nn.Sequential(
            convolutional_net,
            classifier.create_model(opt["classifier_opt"])
        )

    def forward(self, features):
        classification_scores = self.layers(features)
        return classification_scores


def create_model(opt):
    return ConvnetPlusClassifier(opt)
