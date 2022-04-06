import numpy as np
import torch


def feat_extra(fc, pre):  # fc为n个被试的权重矩阵，pre为n次预测结果
    n = fc.shape[0]  # 被试的数目
    m = fc.shape[1]  # 节点的数目

    FW = np.zeros((m, m))  # 初始化关键特征矩阵
    for i in range(0, n):
        p = pre[i]  # 取第i次预测结果
        if p[0] > p[1]:
            prob = p[0]
        else:
            prob = p[1]
        FW = FW + prob * fc[i]  # 循环累加计算关键特征矩阵FW
    FW = FW / n
    # 提取节点集合
    V = np.zeros(m)
    for i in range(0, m):
        for j in range(0, m):
            V[i] = V[i] + FW[i][j]  # 计算各节点重要度
    B = np.argsort(V)
    ver_idx = list(reversed(B))  # B中存储排序后的下标
    ver = sorted(V, reverse=True)  # A中存储排序后的结果
    # 提取边集合
    E = np.zeros(int(m * (m - 1) / 2))
    a = 0
    for i in range(0, m):
        for j in range(i + 1, m):
            E[a] = FW[i][j]
            a += 1
    D = np.argsort(E)
    edg_idx = list(reversed(D))  # D中存储排序后的下标
    edg = sorted(E, reverse=True)  # C中存储排序后的结果


    return ver, ver_idx, edg, edg_idx
