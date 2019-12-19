import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_fully_connected_layer(
    x_in, weight, scale=None, bias=None, normalize_x=True, normalize_w=True
):
    # x_in: a 2D tensor with shape [batch_size x num_features_in]
    # weight: a 2D tensor with shape [num_features_in x num_features_out]
    # scale: (optional) a scalar value
    # bias: (optional) a 1D tensor with shape [num_features_out]

    assert x_in.dim() == 2
    assert weight.dim() == 2
    assert x_in.size(1) == weight.size(0)

    if normalize_x:
        x_in = F.normalize(x_in, p=2, dim=1, eps=1e-12)

    if normalize_w:
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)

    x_out = torch.mm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale.view(1, -1)

    if bias is not None:
        x_out = x_out + bias.view(1, -1)

    return x_out


def batch_cosine_fully_connected_layer(x_in, weight, scale=None, bias=None):
    """
    Args:
        x_in: a 3D tensor with shape
            [meta_batch_size x num_examples x num_features_in]
        weight: a 3D tensor with shape
            [meta_batch_size x num_features_in x num_features_out]
        scale: (optional) a scalar value
        bias: (optional) a 1D tensor with shape [num_features_out]

    Returns:
        x_out: a 3D tensor with shape
            [meta_batch_size x num_examples x num_features_out]
    """

    assert x_in.dim() == 3
    assert weight.dim() == 3
    assert x_in.size(0) == weight.size(0)
    assert x_in.size(2) == weight.size(1)

    x_in = F.normalize(x_in, p=2, dim=2, eps=1e-12)
    weight = F.normalize(weight, p=2, dim=1, eps=1e-12)

    x_out = torch.bmm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale

    if bias is not None:
        x_out = x_out + bias

    return x_out


def global_pooling(x, pool_type):
    assert x.dim() == 4
    if pool_type == "max":
        return F.max_pool2d(x, (x.size(2), x.size(3)))
    elif pool_type == "avg":
        return F.avg_pool2d(x, (x.size(2), x.size(3)))
    else:
        raise ValueError("Unknown pooling type.")


class GlobalPooling(nn.Module):
    def __init__(self, pool_type):
        super().__init__()
        assert pool_type == "avg" or pool_type == "max"
        self.pool_type = pool_type

    def forward(self, x):
        return global_pooling(x, pool_type=self.pool_type)
