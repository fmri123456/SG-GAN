import gc
import os
import scipy.io as io

import feat_extraction

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 不用gpu，cuda有点问题
from torch import nn, tensor

import utils_graph

import time
import torch
import load_data
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from Produce_model import GCN
from Judge_model import JUG_model
import matplotlib.pyplot as plt
import numpy as np
import math
import math

# 加载数据，即：470个样本的脑区——基因网络
train_wei = load_data.load()
train_wei = torch.tensor(train_wei, dtype=torch.float32, device='cuda')
net_size = train_wei.size()  # NCVSAD的net_size[0]=470, net_size[1]=161, net_size[2]=161

# 学习率定为
LR = 0.00001
EPOCH = 100
batch_size = 10
acc_list1 = []
acc_list2 = []
model = GCN(net_size[1], batch_size).cuda()  # 生成器的卷积核和反卷积核的大小为161*161
# 生成器的优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# 生成器的损失函数，公式见详细设计的（3-11）
def loss_func(original_w, predict_w):
    loss_sum = torch.zeros(size=(1, 1), device='cuda')
    for i in range(net_size[1]):
        for j in range(net_size[1]):
            median = math.sqrt(abs(original_w[i][j] - predict_w[i][j]))
            loss_sum = loss_sum + median
            loss_sum = torch.as_tensor(loss_sum, dtype=torch.float32)
            loss_sum = loss_sum.requires_grad_(True)
    return loss_sum


with open("Produce_w.txt", "ab") as file:  # 每跑一次，这个Produce_w.txt需要自行定义
    for epoch in range(EPOCH):
        for step in range(net_size[0]):
            output = model.forward(train_wei[step])
            if epoch == EPOCH - 1:
                result = output.cpu().detach().numpy()
                np.savetxt(file, result)
            print(epoch, '-', step)
            loss1 = loss_func(train_wei[step], output)
            # 清除此训练步骤的梯度
            optimizer.zero_grad()
            # 反向传播，计算梯度
            loss1.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'models.pth')

pro_weight = np.loadtxt("Produce_w.txt")
pro_weight = pro_weight.reshape((net_size[0], net_size[1], net_size[1]))
pro_weight = torch.tensor(pro_weight, dtype=torch.float32, device='cuda')
# 合并网络
Total_net = torch.cat([train_wei, pro_weight], dim=0)  # 940个网络，470个真实+470个生成
Jmodel = JUG_model(net_size[1], batch_size).cuda()  # 判定器的卷积核为161*161

# 判定器的网络标签和患者标签
Net_lable = np.concatenate((np.ones(net_size[0]), np.zeros(net_size[0])))

# NCVSAD的网络标签和患者标签
P_lable = np.concatenate((np.ones(233), np.zeros(237), np.ones(233), np.zeros(237)))
# EMCIVSLMCI的网络标签和患者标签
# P_lable = np.concatenate((np.ones(197), np.zeros(203), np.ones(197), np.zeros(203)))
# LMCIVSAD的网络标签和患者标签
# P_lable = np.concatenate((np.ones(203), np.zeros(233), np.ones(203), np.zeros(233)))

shuffle_idx = np.array(range(0, 2 * net_size[0]))
# 打乱标签
np.random.shuffle(shuffle_idx)
labels1 = P_lable[shuffle_idx]
labels2 = Net_lable[shuffle_idx]
Net_w = Total_net[shuffle_idx]
del Net_lable, P_lable, Total_net
# 分成测试集和训练集
train_id = range(0, 100)
test_id = range(2*net_size[0]-100, 2*net_size[0])
# 网络的测试集和训练集
net_train = Net_w[train_id]
net_test = Net_w[test_id]

# 标签的测试集和训练集
Sick_lable_train = labels1[train_id]
Sick_lable_test = labels1[test_id]
net_lable_train = labels2[train_id]
net_lable_test = labels2[test_id]

Sick_lable_train = utils_graph.onehot_encode(Sick_lable_train)
Sick_lable_train = torch.LongTensor(np.where(Sick_lable_train)[1])
Sick_lable_train = Sick_lable_train.to(device='cuda')

Sick_lable_test = utils_graph.onehot_encode(Sick_lable_test)
Sick_lable_test = torch.LongTensor(np.where(Sick_lable_test)[1])
Sick_lable_test = Sick_lable_test.to(device='cuda')

net_lable_train = utils_graph.onehot_encode(net_lable_train)
net_lable_train = torch.LongTensor(np.where(net_lable_train)[1])
net_lable_train = Sick_lable_train.to(device='cuda')

net_lable_test = utils_graph.onehot_encode(net_lable_test)
net_lable_test = torch.LongTensor(np.where(net_lable_test)[1])
net_lable_test = net_lable_test.to(device='cuda')

# 判定器的优化器
P_optimizer = torch.optim.Adam(Jmodel.parameters(), lr=LR)

# 判定是否为患者以及生成网络的损失函数：交叉熵函数
loss_SN = nn.CrossEntropyLoss().cuda()
dataset = Data.TensorDataset(net_train, net_lable_train, Sick_lable_train)
train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

print('start train...')
if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step, (b_w, nn_lable, ss_lable) in enumerate(train_loader):
            print(step)
            output_lable, output_net = Jmodel.forward(b_w)
            loss1 = loss_SN(output_lable, ss_lable)
            loss2 = loss_SN(output_net, nn_lable)
            loss = loss2 + loss1
            print('loss', loss)
            P_optimizer.zero_grad(set_to_none=True)
            print('P_optimizer...over')
            loss.backward()
            print('loss.backward...over')
            P_optimizer.step()
            print('P_optimizer.step...over')

        # 每10个,测试一次
        if epoch % 10 == 0:
            dataset1 = Data.TensorDataset(net_test, Sick_lable_test, net_lable_test)
            test_loader1 = Data.DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=False)
            Jmodel.eval()
            # 用来训练，训练完后汇总所有的输出lable
            for step, (b, ssick_lable, nnet_lable) in enumerate(test_loader1):
                output_lable, output_net = Jmodel.forward(b)
                if step == 0:
                    Sick_lable_output = output_lable
                    net_lable_output = output_net
                else:
                    Sick_lable_output = torch.cat((Sick_lable_output, output_lable))
                    net_lable_output = torch.cat((net_lable_output, output_net))
            # 提取关键点和边
            pre = Sick_lable_output.cpu().detach().numpy()
            fc = net_test.cpu().detach().numpy()
            ver, ver_idx, edg, edg_idx = feat_extraction.feat_extra(fc, pre)

            TP, TN, FN, FP = utils_graph.stastic_indicators(Sick_lable_output, Sick_lable_test)
            ACC1 = torch.true_divide((TP + TN), (TP + TN + FP + FN))
            ACC1 = ACC1.cpu().detach().numpy()
            SEN1 = torch.true_divide(TP, (TP + FN))
            SEN1 = SEN1.cpu().detach().numpy()
            SPE1 = torch.true_divide(TN, (FP + TN))
            SPE1 = SPE1.cpu().detach().numpy()
            BAC1 = torch.true_divide((SEN1 + SPE1), 2)
            BAC1 = BAC1.cpu().detach().numpy()
            acc_list1.append(ACC1)

            TP, TN, FN, FP = utils_graph.stastic_indicators(net_lable_output, net_lable_test)
            ACC2 = torch.true_divide((TP + TN), (TP + TN + FP + FN))
            ACC2 = ACC2.cpu().detach().numpy()
            SEN2 = torch.true_divide(TP, (TP + FN))
            SEN2 = SEN2.cpu().detach().numpy()
            SPE2 = torch.true_divide(TN, (FP + TN))
            SPE2 = SPE2.cpu().detach().numpy()
            BAC2 = torch.true_divide((SEN2 + SPE2), 2)
            BAC2 = BAC2.cpu().detach().numpy()
            acc_list2.append(ACC2)

    torch.save(Jmodel.state_dict(), 'Judge_model.pth')

    x = range(1, 10)
    # 画分类准确率图
    plt.figure(num=2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(x, acc_list1)
    plt.show()

    # 画分类准确率图
    plt.figure(num=2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(x, acc_list2)
    plt.show()
    np.save('sick_accuracy.npy', acc_list1)
    np.save('net_accuracy.npy', acc_list2)
    np.save('ACC1.npy', ACC1)
    np.save('SEN1.npy', SEN1)
    np.save('SPE1.npy', SPE1)
    np.save('BAC1.npy', BAC1)

    np.save('ACC2.npy', ACC2)
    np.save('SEN2.npy', SEN2)
    np.save('SPE2.npy', SPE2)
    np.save('BAC2.npy', BAC2)

    np.save('ver.npy', ver)
    np.save('ver_idx.npy', ver_idx)
    np.save('edg.npy', edg)
    np.save('edg_idx.npy', edg_idx)
