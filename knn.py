# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月08日
"""

import torch
import torch.nn as nn


def knn_with_explanation(x, k):
    print("knn_init:")
    print(x.shape)
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)  # [b,num,num]
    print("transpose:")
    print(x.transpose(2, 1).shape)
    print("inner:")
    print(inner.shape)
    # 求坐标（维度空间）的平方和
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b,1,num] #x ** 2 表示点平方而不是x*x
    print("xx:")
    print(xx.shape)
    # 2x1x2+2y1y2+2z1z2-x1^2-y1^2-z1^2-x2^2-y2^2-z2^2=-[(x1-x2)^2+(y1-y2)^2+(z1-z2)^2]
    pairwise_distance = -xx - inner
    del inner, x
    pairwise_distance = pairwise_distance - xx.transpose(2, 1)  # [b,num,num]
    print("pairwise_distance:")
    print(pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    print(pairwise_distance)
    print(pairwise_distance.topk(k=k, dim=-1))
    print("*"*20)
    print(pairwise_distance.topk(k=k, dim=-1)[1])
    print("idx:")
    print(idx.shape)
    print("knn_out:")
    print(idx)
    return idx


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)  # [b,num,num]
    # 求坐标（维度空间）的平方和
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b,1,num] #x ** 2 表示点平方而不是x*x
    # 2x1x2+2y1y2+2z1z2-x1^2-y1^2-z1^2-x2^2-y2^2-z2^2=-[(x1-x2)^2+(y1-y2)^2+(z1-z2)^2]
    pairwise_distance = -xx - inner
    del inner, x
    pairwise_distance = pairwise_distance - xx.transpose(2, 1)  # [b,num,num]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


batch_size = 8
num_dims = 64
num_points = 10
k2 = 20
# k = 7
# a = torch.zeros((batch_size, num_dims, num_points))
# c = torch.tensor([[[1, 2, 5, 8, 9, 4, 3, 2],
#                    [0, 6, 9, 2, 4, 6, 7, 9],
#                    [1, 5, 6, 8, 7, 3, 4, 6],
#                    [5, 7, 9, 0, 1, 3, 4, 8]]])
# print(c.shape)
# b = knn(a, k)
# print((torch.arange(0, batch_size).view(-1, 1, 1) * num_points + b).view(-1).shape)
act_f = nn.LeakyReLU(negative_slope=0.01, inplace=True)
in_put = torch.randn(batch_size, num_dims * 2, num_points, k2)
print(in_put.shape)
convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), act_f)
convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), act_f)
convSN1 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), act_f)
output1 = convDG1(in_put)
x1 = output1.max(dim=-1, keepdim=True)[0]
print(x1.shape)
output2 = convDG2(output1)
print(output1.shape)
print(output2.shape)
