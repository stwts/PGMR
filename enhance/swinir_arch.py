import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoin
#from .arch_util import to_2tuple, trunc_normal_
from functools import partial
from importlib import import_module
import os
# from GaussianSmoothLayer import GaussionSmoothLayer
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class _Residual_Block(nn.Module):
    def __init__(self, n_feat=64, reduction=8):
        super(_Residual_Block, self).__init__()
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.in1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, )
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    #  self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))

        output = ((self.conv2(output)))
        # sa_branch = self.SA(output)
        # ca_branch = self.CA(output)
        # res = torch.cat([sa_branch, ca_branch], dim=1)
        # res = self.conv1x1(res)

        output = torch.add(self.relu(output), identity_data)
        # output = torch.add(output,identity_data)

        return output



class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # self.FeatureExtractor = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     # nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2, stride=2),
        #     # nn.Linear(64 * 64 * 64, 256),  # 适应你的特征维度
        # )
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, ),
        )
        norm = spectral_norm
        # downsample
        self.conv1 = norm(nn.Conv2d(64, 64, 4, 2, 1, bias=False))
        # self.conv2 = norm(nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1, bias=False))
        # self.conv3 = norm(nn.Conv2d(128 * 4, 128 * 8, 4, 2, 1, bias=False))
        # upsample
        # self.conv4 = norm(nn.Conv2d(128 * 8, 128 * 4, 3, 1, 1, bias=False))
        # self.conv5 = norm(nn.Conv2d(128 * 4, 128 * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(64, 64, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(64, 64, 3, 1, 1, bias=False))

        self.residual = self.make_layer(_Residual_Block, 6)
        # self.dab = self.make_layer(DAB, 6)
        self.conv_output = nn.Sequential(
            nn.Conv2d(in_channels=64 , out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, ),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3, ),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, ),

        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3, ),
        )
       
        # self.downsample_conv_2x = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        # self.downsample_conv_3x = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=3, padding=1)
        # self.downsample_conv_4x = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=4, padding=1)
        # self.downsample_conv_8x = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=8, padding=1)

        self.downsample_conv_1x = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.downsample_conv_2x = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3)
        self.downsample_conv_3x = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=3, padding=3)
        self.downsample_conv_4x = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=4, padding=3)

        # self.scale = nn.Parameter(torch.randn(3, 1, 1), requires_grad=True)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    

    

    def forward(self, x):
        res = x

        x = self.conv_input(x)

        out = self.residual(x)

        out = self.conv_output(out)

        out = out + self.conv_fusion(x)
        out = out + x

        out =self.downsample_conv_1x(out)
        out = out + res

        return out




