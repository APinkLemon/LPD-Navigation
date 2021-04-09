# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月08日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TranformNet(nn.Module):
    def __init__(self, k=3, negative_slope=1e-2, use_relu=True):
        super(TranformNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        if use_relu:
            self.relu = nn.ReLU
        else:
            self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)), inplace=True)
        x = F.relu(self.bn5(self.fc2(x)), inplace=True)
        x = self.fc3(x)

        device = torch.device('cuda')

        iden = torch.eye(self.k, dtype=torch.float32, device=device).view(1, self.k * self.k).repeat(batchsize, 1)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
