import torch.nn as nn
import torchvision.models.resnet as resnet_utils


class ResNetBlock(nn.Module):
    def __init__(self, block_type, inplanes, planes, num_layers, stride):
        super().__init__()

        if block_type == "BasicBlock":
            block = resnet_utils.BasicBlock
        elif block_type == "Bottleneck":
            block = resnet_utils.Bottleneck
        else:
            raise ValueError(f"Invalid block type {block_type}")

        downsample = None
        if stride != 1 or inplanes != (planes * block.expansion):
            conv1x1 = nn.Conv2d(
                inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False
            )
            downsample = nn.Sequential(conv1x1, nn.BatchNorm2d(planes * block.expansion))

        layers = [
            block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for _ in range(1, num_layers):
            layers.append(block(inplanes=inplanes, planes=planes))

        self.resnet_block = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.resnet_block(x)


def create_model(opt):
    return ResNetBlock(
        block_type=opt["block_type"],
        inplanes=opt["inplanes"],
        planes=opt["planes"],
        num_layers=opt["num_layers"],
        stride=opt["stride"],
    )
