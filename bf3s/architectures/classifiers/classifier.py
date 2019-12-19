import numpy as np
import torch.nn as nn

import bf3s.architectures.classifiers.utils as cutils
import bf3s.architectures.tools as tools


class Classifier(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.classifier_type = opt["classifier_type"]
        self.num_channels = opt["num_channels"]
        self.num_classes = opt["num_classes"]
        self.global_pooling = opt["global_pooling"] if ("global_pooling" in opt) else False

        if self.classifier_type == "cosine":
            bias = opt["bias"] if ("bias" in opt) else False
            self.layers = cutils.CosineClassifier(
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                scale=opt["scale_cls"],
                learn_scale=opt["learn_scale"],
                bias=bias,
            )

        elif self.classifier_type == "linear":
            bias = opt["bias"] if ("bias" in opt) else True
            self.layers = nn.Linear(self.num_channels, self.num_classes, bias=bias)
            if bias:
                self.layers.bias.data.zero_()

            fout = self.layers.out_features
            self.layers.weight.data.normal_(0.0, np.sqrt(2.0 / fout))

        elif self.classifier_type == "mlp_linear":
            mlp_channels = opt["mlp_channels"]
            num_mlp_channels = len(mlp_channels)
            mlp_channels = [self.num_channels,] + mlp_channels
            self.layers = nn.Sequential()

            pre_act_relu = opt["pre_act_relu"] if ("pre_act_relu" in opt) else False
            if pre_act_relu:
                self.layers.add_module("pre_act_relu", nn.ReLU(inplace=False))

            for i in range(num_mlp_channels):
                self.layers.add_module(
                    f"fc_{i}", nn.Linear(mlp_channels[i], mlp_channels[i + 1], bias=False),
                )
                self.layers.add_module(f"bn_{i}", nn.BatchNorm1d(mlp_channels[i + 1]))
                self.layers.add_module(f"relu_{i}", nn.ReLU(inplace=True))

            fc_prediction = nn.Linear(mlp_channels[-1], self.num_classes)
            fc_prediction.bias.data.zero_()
            self.layers.add_module("fc_prediction", fc_prediction)
        else:
            raise ValueError(
                "Not implemented / recognized classifier type {}".format(self.classifier_type)
            )

    def flatten(self):
        return (
            self.classifier_type == "linear"
            or self.classifier_type == "cosine"
            or self.classifier_type == "mlp_linear"
        )

    def forward(self, features):
        if self.global_pooling:
            features = tools.global_pooling(features, pool_type="avg")

        if features.dim() > 2 and self.flatten():
            features = features.view(features.size(0), -1)

        scores = self.layers(features)
        return scores


def create_model(opt):
    return Classifier(opt)
