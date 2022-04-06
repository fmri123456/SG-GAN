import gc
import math
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F

class Deconvolution_layer(nn.Module):
    def __init__(self, kernel_size, net_sum, use_bias=True):
        super(Deconvolution_layer, self).__init__()
        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.net_sum = net_sum
        # 反卷积核
        self.kernel = Parameter(torch.FloatTensor(kernel_size, kernel_size))
        self.reset_parameters()

    def save_weight(self):
        pass

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kernel.size(1))
        self.kernel.data.uniform_(-stdv, stdv)

    def forward1(self, input_feature):
        kernel = self.kernel
        kernel = kernel.to(device='cuda')
        output = torch.zeros(size=(self.kernel_size, self.kernel_size), device='cuda')
        # 反卷积的公式实现，公式见详细设计的（3-7）
        support_1 = torch.matmul(input_feature, kernel)
        support_2 = torch.matmul(kernel, input_feature)
        for r in range(self.kernel_size):
            for c in range(self.kernel_size):
                output[r, c] = support_1[r, r] + support_2[c, c]
        del support_2, support_1, kernel

        if self.use_bias:
            output += self.bias
        output = F.relu(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
