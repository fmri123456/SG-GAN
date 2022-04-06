# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import torch

from layer import Convolution_layer
import torch.nn as nn
import torch.nn.functional as F

from layer_back import Deconvolution_layer


# 判定器模型设计
class JUG_model(nn.Module):
    def __init__(self, kernel_size, net_sum):
        super(JUG_model, self).__init__()  # 初始化
        self.kernel_size = kernel_size
        self.net_sum = net_sum
        # 卷积层
        self.conv1 = Convolution_layer(kernel_size, net_sum, False).cuda()
        # 连接层设计
        self.fc1 = nn.Linear(kernel_size * kernel_size, kernel_size // 2).cuda()
        self.fc2 = nn.Linear(kernel_size // 2, 2).cuda()

    def forward(self, input_feature):
        # 卷积层后的结果
        cn1 = F.relu(self.conv1.forward(input_feature))
        # 拉直运算
        cn2_rl = cn1.reshape(-1, 161 * cn1.shape[1])
        # 全连接层运算
        fc1 = F.relu(self.fc1(cn2_rl))
        judge_label = F.softmax(self.fc2(fc1))  # 判断是否为患者
        judge_net = F.sigmoid(self.fc2(fc1))  # 判断网络是否为生成的
        return judge_label, judge_net
