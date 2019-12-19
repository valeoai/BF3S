import numpy as np
import torch
import torch.nn as nn

import bf3s.architectures.tools as tools


class CosineClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        scale=20.0,
        learn_scale=False,
        bias=False,
        normalize_x=True,
        normalize_w=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize_x = normalize_x
        self.normalize_w = normalize_w

        weight = torch.FloatTensor(num_classes, num_channels).normal_(
            0.0, np.sqrt(2.0 / num_channels)
        )
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        scale_cls = torch.FloatTensor(1).fill_(scale)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=learn_scale)

    def forward(self, x_in):
        assert x_in.dim() == 2
        return tools.cosine_fully_connected_layer(
            x_in,
            self.weight.t(),
            scale=self.scale_cls,
            bias=self.bias,
            normalize_x=self.normalize_x,
            normalize_w=self.normalize_w,
        )

    def extra_repr(self):
        s = "num_channels={}, num_classes={}, scale_cls={} (learnable={})".format(
            self.num_channels,
            self.num_classes,
            self.scale_cls.item(),
            self.scale_cls.requires_grad,
        )
        learnable = self.scale_cls.requires_grad
        s = (
            f"num_channels={self.num_channels}, "
            f"num_classes={self.num_classes}, "
            f"scale_cls={self.scale_cls.item()} (learnable={learnable}), "
            f"normalize_x={self.normalize_x}, normalize_w={self.normalize_w}"
        )

        if self.bias is None:
            s += ", bias=False"
        return s


def average_train_features(features_train, labels_train):
    labels_train_transposed = labels_train.transpose(1, 2)
    weight_novel = torch.bmm(labels_train_transposed, features_train)
    weight_novel = weight_novel.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel)
    )

    return weight_novel


def preprocess_5D_features(features, global_pooling):
    meta_batch_size, num_examples, channels, height, width = features.size()
    features = features.view(meta_batch_size * num_examples, channels, height, width)

    if global_pooling:
        features = tools.global_pooling(features, pool_type="avg")

    features = features.view(meta_batch_size, num_examples, -1)

    return features
