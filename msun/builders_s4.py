import copy
import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from typing import List, Tuple

# Layer counts for MobileNetV2 and VGG16
LAYERS_MOBILENET = 10
LAYERS_VGG = 24


def build_resnet50(res_lists: List[List[int]], base: nn.Module, device):
    """Build unified head and per-scale subnets for ResNet50."""
    u = copy.deepcopy(base)
    u.conv1 = nn.Identity()
    u.bn1 = nn.Identity()
    u.relu = nn.Identity()
    u.maxpool = nn.Identity()
    u.layer1 = nn.Identity()

    # build subnes
    configs = [
        {'k': 3, 's': 1, 'p': 2, 'pool': False, 'r': res_lists[0]},
        {'k': 3, 's': 1, 'p': 2, 'pool': False, 'r': res_lists[1]},
        {'k': 5, 's': 1, 'p': 2, 'pool': True, 'r': res_lists[2]},
        {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[3]},
        {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[4]}
    ]
    res_lists = [c['r'] for c in configs]
    subnets = nn.ModuleList()
    for c in configs:
        layers = [nn.Conv2d(3, 64, c['k'], c['s'], c['p'], bias=False),
                  nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        if c['pool']:
            layers.append(nn.MaxPool2d(3, 2, 1))
        layers.append(copy.deepcopy(base.layer1))
        subnets.append(nn.Sequential(*layers))

    # Automatically determine the unified spatial size from the last subnet
    with torch.no_grad():
        max_res = max(res_lists[-1])
        dummy = torch.zeros(1, 3, max_res, max_res, device=device)
        z_size = subnets[-1](dummy).shape[-1]

    return u, subnets, res_lists, z_size


def build_densenet101(res_lists: List[List[int]], base: nn.Module, device):
    # Unified head: remove initial conv/pool and first denseblock
    u = copy.deepcopy(base)
    u.features.conv0 = nn.Identity()
    u.features.norm0 = nn.Identity()
    u.features.relu0 = nn.Identity()
    u.features.pool0 = nn.Identity()
    u.features.denseblock1 = nn.Identity()
    u.features.transition1 = nn.Identity()
    # u.features.denseblock2 = nn.Identity()
    # u.features.transition2 = nn.Identity()
    unified_net = u

    # Per-scale subnets: initial conv layers + first denseblock
    configs = [
        {'k': 3, 's': 1, 'p': 1, 'pool': False, 'r': res_lists[0]},
        {'k': 3, 's': 1, 'p': 1, 'pool': False, 'r': res_lists[1]},
        {'k': 5, 's': 1, 'p': 2, 'pool': True, 'r': res_lists[2]},
        {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[3]},
        {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[4]},
    ]
    res_lists = [c['r'] for c in configs]
    subnets = nn.ModuleList()
    for c in configs:
        layers = [
            nn.Conv2d(3, 64, kernel_size=c['k'], stride=c['s'], padding=c['p'], bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        if c['pool']:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # attach DenseNet first block
        layers.append(copy.deepcopy(base.features.denseblock1))
        layers.append(copy.deepcopy(base.features.transition1))
        # layers.append(copy.deepcopy(base.features.denseblock2))
        # layers.append(copy.deepcopy(base.features.transition2))
        subnets.append(nn.Sequential(*layers))

    # Determine spatial size for unified head
    with torch.no_grad():
        max_res = max(res_lists[-1])
        dummy = torch.zeros(1, 3, max_res, max_res, device=device)
        z_size = subnets[-1](dummy).shape[-1]

    return u, subnets, res_lists, z_size


def build_vgg16(res_lists: List[List[int]], base: nn.Module, device):
    """Build unified head and per-scale subnets for VGG16-BN."""
    # Unified head: remove first LAYERS convolutional layers
    u = copy.deepcopy(base)
    for i in range(LAYERS_VGG):
        u.features[i] = nn.Identity()

    # Per-scale subnets with custom pooling modifications
    subnets = nn.ModuleList()
    for idx, _ in enumerate(res_lists):
        v = copy.deepcopy(base)
        layers = []
        for i in range(LAYERS_VGG):
            layer = v.features[i]
            # First subnet: replace MaxPool at layers 6,13,23 with stride-1 pooling
            if idx in [0, 1] and i in [6, 13, 23]:
                layer = nn.MaxPool2d(kernel_size=2, stride=1)
            # Second subnet: replace MaxPool at layer 23 with stride-1 pooling
            elif idx == 2 and i == 23:
                layer = nn.MaxPool2d(kernel_size=2, stride=1)
            layers.append(layer)
        subnets.append(nn.Sequential(*layers))

    # Determine feature-map spatial size after subnets
    with torch.no_grad():
        max_res = max(res_lists[-1])
        dummy = torch.zeros(1, 3, max_res, max_res, device=device)
        z_size = subnets[-1](dummy).shape[-1]

    return u, subnets, res_lists, z_size


def build_mobilenetv2(res_lists: List[List[int]], base: nn.Module, device):
    """Build unified head and per-scale subnets for MobileNetV2."""
    u = copy.deepcopy(base)
    for i in range(LAYERS_MOBILENET):
        u.features[i] = nn.Identity()

    # subnets for each scale
    subnets = nn.ModuleList()
    for idx in range(len(res_lists)):
        v = copy.deepcopy(base)
        layers = [v.features[i] for i in range(LAYERS_MOBILENET)]
        if idx == 0:
            # first subnet: 1x1 conv and custom depthwise blocks
            layers[0] = Conv2dNormActivation(3, 32, kernel_size=1, stride=1,
                                             norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
            layers[2].conv[1] = Conv2dNormActivation(96, 96, kernel_size=3, stride=1, groups=96,
                                                     norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
            layers[4].conv[1] = Conv2dNormActivation(144, 144, kernel_size=3, stride=1, groups=144,
                                                     norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        elif idx == 1:
            # second subnet: 3x3 first conv
            layers[0] = Conv2dNormActivation(3, 32, kernel_size=3, stride=1,
                                             norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        subnets.append(nn.Sequential(*layers))

    # Determine feature-map spatial size after subnets
    with torch.no_grad():
        max_res = max(res_lists[-1])
        dummy = torch.zeros(1, 3, max_res, max_res, device=device)
        z_size = subnets[-1](dummy).shape[-1]

    return u, subnets, res_lists, z_size
