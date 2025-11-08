import torch
from torch import nn, Tensor
from torch.nn import Module, Sequential
import torchvision


def my_mlp(size: list) -> Sequential:
    layers = list()
    for i in range(len(size) - 1):
        layers.append(nn.Linear(size[i], size[i+1]))
        layers.append(nn.ReLU(True))
    return Sequential(*layers)


def my_resnet(args: tuple) -> Sequential:
    x_size, resnet_num, k_size, stride, padding, y_size = args
    # resnet: 3 7 2 3
    # hashvfl: 1 3 1 1

    if resnet_num == 18:
        f = torchvision.models.resnet18
    elif resnet_num == 34:
        f = torchvision.models.resnet34
    elif resnet_num == 50:
        # f = torchvision.models.resnet50
        f = torchvision.models.resnext50_32x4d
    elif resnet_num == 101:
        f = torchvision.models.wide_resnet101_2
    elif resnet_num == 152:
        f = torchvision.models.resnet152
    else:
        raise ValueError('What is resnet%d' % resnet_num)

    model = f(num_classes=y_size, pretrained=False)

    '''
    ResNet:
    >>> self.conv1 = nn.Conv2d(
            3, self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
    >>> self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    >>> self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    '''

    conv1 = nn.Conv2d(
        x_size, model.conv1.out_channels,
        kernel_size=k_size,
        stride=stride,
        padding=padding,
        bias=False
    )

    avgpool = nn.AvgPool2d(4, padding=1)
    flatten = nn.Flatten()

    return Sequential(
        conv1, model.bn1, model.relu,
        model.layer1, model.layer2,
        model.layer3,
        model.layer4, avgpool, flatten, model.fc
    )
