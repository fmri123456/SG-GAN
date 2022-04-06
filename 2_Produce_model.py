# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import torch

from layer import Convolution_layer
import torch.nn as nn
import torch.nn.functional as F

from layer_back import Deconvolution_layer


# 生成器模型设计
class GCN(nn.Module):
    def __init__(self, kernel_size, net_sum):
        super(GCN, self).__init__()  # 初始化
        self.kernel_size = kernel_size
        self.net_sum = net_sum
        # 卷积层
        self.conv1 = Convolution_layer(kernel_size, net_sum, False).cuda()
        # 反卷积层,
        self.Deconv1 = Deconvolution_layer(kernel_size, net_sum, False).cuda()

    def forward(self, input_feature):
        # 卷积层后的结果
        cn1 = F.relu(self.conv1.forward1(input_feature))
        # 反卷积后的结果
        Dcn1 = F.relu(self.Deconv1.forward1(cn1))
        return Dcn1