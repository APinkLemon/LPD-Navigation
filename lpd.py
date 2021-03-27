# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月08日
"""

import torch
from config import cfg
from utils import knn, TranformNet
import torch.nn as nn
import torch.nn.functional as F


# True -> cat, False -> stack.
cat_or_stack = cfg.net.cat_or_stack


def get_graph_feature_Origin(x, k=20, idx=None, cat = True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    device = torch.device('cuda')
    # 获得索引阶梯数组
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1,
                                                               1) * num_points  # (batch_size, 1, 1) [0 num_points ... num_points*(B-1)]
    # 以batch为单位，加到索引上
    idx = idx + idx_base  # (batch_size, num_points, k)
    # 展成一维数组，方便后续索引
    idx = idx.view(-1)  # (batch_size * num_points * k)
    # 获得特征维度
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    # 改变x的shape，方便索引。被索引数组是所有batch的所有点的特征，索引数组idx为所有临近点对应的序号，从而索引出所有领域点的特征
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k,num_dims)
    # 统一数组形式
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    if cat:
        # 重复k次，以便k个邻域点每个都能和中心点做运算
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [B, num, k, num_dims]
        # 领域特征的表示，为(feature - x, x)，这种形式可以详尽参见dgcnn论文
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)  # [B, num_dims*2, num, k]
    else:
        feature = feature.permute(0, 3, 1, 2)
    return feature


class LPDNetOrign(nn.Module):
    def __init__(self, emb_dims=512, use_mFea=False, t3d=True, tfea=False, use_relu=False):
        super(LPDNetOrign, self).__init__()
        self.negative_slope = 1e-2
        if use_relu:
            self.act_f = nn.ReLU(inplace=True)
        else:
            self.act_f = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
        self.use_mFea = use_mFea
        if self.use_mFea:
            initFeaNum = 8
        else:
            initFeaNum = 3
        self.k = 20
        self.t3d = t3d
        self.tfea = tfea
        self.emb_dims = emb_dims
        if self.t3d:
            self.t_net3d = TranformNet(3)
        if self.tfea:
            self.t_net_fea = TranformNet(64)
        self.useBN = True
        if self.useBN:
            # [b,6,num,20] 输入 # 激活函数换成Leaky ReLU? 因为加了BN，所以bias可以舍弃
            self.convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64),self.act_f)
            self.convDG2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64),self.act_f)
            self.convSN1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64),self.act_f)
            self.convSN2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64),self.act_f)
            # 在一维上进行卷积，临近也是左右概念，类似的，二维卷积，临近有上下左右的概念 # 在relu之前进行batchNorm避免梯度消失，同时使分布不一直在变化
            self.conv1_lpd = nn.Sequential(nn.Conv1d(initFeaNum, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), self.act_f)
            self.conv2_lpd = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), self.act_f)
            self.conv3_lpd = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), self.act_f)
            self.conv4_lpd = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128), self.act_f)
            self.conv5_lpd = nn.Sequential(nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False), nn.BatchNorm1d(self.emb_dims), self.act_f)
        else:
            # [b,6,num,20] 输入 # 激活函数换成Leaky ReLU? 因为加了BN，所以bias可以舍弃
            self.convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=True),self.act_f)
            self.convDG2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),self.act_f)
            self.convSN1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),self.act_f)
            self.convSN2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),self.act_f)
            # 在一维上进行卷积，临近也是左右概念，类似的，二维卷积，临近有上下左右的概念 # 在relu之前进行batchNorm避免梯度消失，同时使分布不一直在变化
            self.conv1_lpd = nn.Sequential(nn.Conv1d(initFeaNum, 64, kernel_size=1, bias=True), self.act_f)
            self.conv2_lpd = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=True), self.act_f)
            self.conv3_lpd = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=True), self.act_f)
            self.conv4_lpd = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=True), self.act_f)
            self.conv5_lpd = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=True), self.act_f)
    # input x: # [B,1,num,num_dims]
    # output x: # [b,emb_dims,num,1]
    def forward(self, x):
        x = torch.squeeze(x, dim=1).transpose(2, 1)  # [B,num_dims,num]
        batch_size, num_dims, num_points = x.size()
        # 单独对坐标进行T-Net旋转
        if num_dims > 3 or self.use_mFea:
            x, feature = x.transpose(2, 1).split([3, 5], dim=2)  # [B,num,3]  [B,num,num_dims-3]
            xInit3d = x.transpose(2, 1)
            # 是否进行3D坐标旋转
            if self.t3d:
                trans = self.t_net3d(x.transpose(2, 1))
                x = torch.bmm(x, trans)
                x = torch.cat([x, feature], dim=2).transpose(2, 1)  # [B,num_dims,num]
            else:
                x = torch.cat([x, feature], dim=2).transpose(2, 1)  # [B,num_dims,num]
        else:
            xInit3d = x
            if self.t3d:
                trans = self.t_net3d(x)
                x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)

        x = self.conv1_lpd(x)
        x = self.conv2_lpd(x)

        if self.tfea:
            trans_feat = self.t_net_fea(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        # Serial structure
        # Danymic Graph cnn for feature space
        x = get_graph_feature_Origin(x, k=self.k)  # [b,64*2,num,20]
        x = self.convDG1(x)  # [b,64,num,20]
        x = self.convDG2(x)  # [b,64,num,20]
        x = x.max(dim=-1, keepdim=True)[0]  # [b,64,num,1]

        # Spatial Neighborhood fusion for cartesian space
        idx = knn(xInit3d, k=self.k)
        x = get_graph_feature_Origin(x, idx=idx, k=self.k, cat=False)  # [b,64,num,20]
        x = self.convSN1(x)  # [b,64,num,20]
        x = self.convSN2(x)  # [b,64,num,20]
        x = x.max(dim=-1, keepdim=True)[0].squeeze(-1)  # [b,64,num]

        x = self.conv3_lpd(x)  # [b,64,num]
        x = self.conv4_lpd(x)  # [b,128,num]
        x = self.conv5_lpd(x)  # [b,emb_dims,num]
        x = x.unsqueeze(-1) # [b,emb_dims,num,1]

        return x


# input (batch_size, num_dims, num_points)
# output (batch_size, num_points, k, num_dims * 2)
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        # (batch_size, num_points, k)
        idx = knn(x, k=k)

    device = torch.device('cuda')
    # (batch_size, 1, 1) [0, num_points, ..., num_points * (batch_size - 1)]
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # (batch_size, num_points, k)
    idx = idx + idx_base
    # (batch_size * num_points * k)
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    # (batch_size, num_points, num_dims)
    x = x.transpose(2, 1).contiguous()
    # (batch_size * num_points * k, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # (batch_size, num_points, k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)
    '''
    feature: (batch_size, num_points, k, num_dims)
    For every batch, here are points.
    For every point, here are k nearest points.
    For every point, here are dims.
    '''

    if cat_or_stack:
        # (batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        # (batch_size, num_points, k, num_dims * 2)
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
        '''
        feature: (batch_size, num_dims * 2, num_points, k)
        '''
    else:
        # (batch_size, num_points, 1, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims)
        # (batch_size, num_points, k + 1, num_dims)
        feature = torch.cat((feature, x), dim=2).permute(0, 3, 1, 2)
        '''
        feature: (batch_size, num_dims, num_points, k + 1)
        '''
    return feature


# input: (batch_size, 1, num_points, num_dims)
# output: (batch_size, self.emb_dims, num_points, 1)
class LPDNet(nn.Module):
    def __init__(self, emb_dims=512, use_mFea=False, t3d=True, tfea=False, use_relu=False, negative_slope=1e-2):
        super(LPDNet, self).__init__()
        self.k = 20
        self.t3d = t3d
        self.tfea = tfea
        self.useBN = True
        self.use_mFea = use_mFea
        self.emb_dims = emb_dims
        self.negative_slope = negative_slope
        if use_relu:
            self.act_f = nn.ReLU(inplace=True)
        else:
            self.act_f = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
        '''
        Get the input hyper param. 
        '''

        if self.t3d:
            self.t_net3d = TranformNet(3)
        if self.tfea:
            self.t_net_fea = TranformNet(64)
        '''
        Set T-Net.  
        '''

        if self.useBN:
            # 激活函数换成Leaky ReLU. 因为加了BN，所以bias可以舍弃.
            if cat_or_stack:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256),
                                             self.act_f)
            else:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 1, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 1, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256),
                                             self.act_f)
            '''
            Set first convDG: 
            Dim1: 64 * 2 -> 128. 
            Set second convDG: 
            Dim1: 128 -> 128. 
            Set first convSN:
            Dim1: 128 -> 256.
            '''

            if self.use_mFea:
                self.conv1_lpd = nn.Conv1d(8, 64, kernel_size=1, bias=False)
            else:
                self.conv1_lpd = nn.Conv1d(3, 64, kernel_size=1, bias=False)
            '''
            Set first conv1d: 
            Dim1: 3 or 8 -> 64. 
            '''

            self.conv2_lpd = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            '''
            Set second conv1d: 
            Dim1: 64 -> 64. 
            '''

            self.conv3_lpd = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False)
            '''
            Set third conv1d:
            Dim1: 512 -> self.emb_dims. 
            '''

            # 在 Relu 之前进行 BatchNorm 避免梯度消失，同时使分布不一直在变化
            self.bn1_lpd = nn.BatchNorm1d(64)
            self.bn2_lpd = nn.BatchNorm1d(64)
            self.bn3_lpd = nn.BatchNorm1d(self.emb_dims)
            '''
            Set batch norm.
            '''
        else:
            if cat_or_stack:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=True), self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True), self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=True), self.act_f)
            else:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 1, 128, kernel_size=1, bias=True), self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True), self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 1, 256, kernel_size=1, bias=True), self.act_f)

            if self.use_mFea:
                self.conv1_lpd = nn.Conv1d(8, 64, kernel_size=1, bias=True)
            else:
                self.conv1_lpd = nn.Conv1d(3, 64, kernel_size=1, bias=True)
            self.conv2_lpd = nn.Conv1d(64, 64, kernel_size=1, bias=True)
            self.conv3_lpd = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=True)

    # input: (batch_size, 1, num_points, num_dims)
    # output: (batch_size, self.emb_dims, num_points, 1)
    def forward(self, x):
        # (batch_size, num_dims, num_points)
        x = torch.squeeze(x, dim=1).transpose(2, 1)
        batch_size, num_dims, num_points = x.size()

        if num_dims > 3 or self.use_mFea:
            x, feature = x.transpose(2, 1).split([3, 5], dim=2)  # [B,num,3]  [B,num,num_dims-3]
            xInit3d = x.transpose(2, 1)
            # 是否进行3D坐标旋转
            if self.t3d:
                trans = self.t_net3d(x.transpose(2, 1))
                x = torch.bmm(x, trans)
                x = torch.cat([x, feature], dim=2).transpose(2, 1)  # [B,num_dims,num]
            else:
                x = torch.cat([x, feature], dim=2).transpose(2, 1)  # [B,num_dims,num]
        else:
            xInit3d = x
            if self.t3d:
                # (num_dims, num_dims)
                trans = self.t_net3d(x)
                # (batch_size, num_dims, num_points)
                x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        '''
        Get x updated by T-Net.
        x: (batch_size, num_dims, num_points)
        Get backup of init x.
        xInit3d: (batch_size, num_dims, num_points)
        '''

        if self.useBN:
            x = self.act_f(self.bn1_lpd(self.conv1_lpd(x)))
            x = self.act_f(self.bn2_lpd(self.conv2_lpd(x)))
        else:
            x = self.act_f(self.conv1_lpd(x))
            x = self.act_f(self.conv2_lpd(x))
        '''
        Get x updated by conv1 and conv2.
        x: (batch_size, 64, num_points)
        '''

        if self.tfea:
            trans_feat = self.t_net_fea(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        '''
        Get x updated by T-Net.
        x: (batch_size, num_dims, num_points)
        num_dims = 64
        '''

        # Serial structure
        # Dynamic Graph cnn for feature space
        if cat_or_stack:
            # (batch_size, num_dims * 2, num_points, k)
            x = get_graph_feature(x, k=self.k)
        else:
            # (batch_size, num_dims, num_points, k + 1)
            x = get_graph_feature(x, k=self.k)
        '''
        Get x including local feature.
        x: (batch_size, num_dims * 2, num_points, k)
        num_dims = 64
        '''

        # (batch_size, 128, num_points, k)
        x = self.convDG1(x)
        # (batch_size, 128, num_points, 1)
        x1 = x.max(dim=-1, keepdim=True)[0]
        # (batch_size, 128, num_points, k)
        x = self.convDG2(x)
        # (batch_size, 128, num_points, 1)
        x2 = x.max(dim=-1, keepdim=True)[0]
        '''
        Get x1 and x2.
        x1: (batch_size, num_dims * 2, num_points, 1)
        x2: (batch_size, num_dims * 2, num_points, 1)
        num_dims = 64: (batch_size, 128, num_points, 1)
        '''

        # Spatial Neighborhood fusion for cartesian space
        # (batch_size, num_points, k)
        idx = knn(xInit3d, k=self.k)
        # (batch_size, 128 * 2, num_points, k)
        x = get_graph_feature(x2, idx=idx, k=self.k)
        # (batch_size, 256, num_points, k)
        x = self.convSN1(x)
        # (batch_size, 256, num_points, 1)
        x3 = x.max(dim=-1, keepdim=True)[0]
        '''
        Get x3.
        x3: (batch_size, num_dims * 4, num_points, 1)
        num_points = 64: (batch_size, 256, num_points, 1)
        '''

        # (batch_size, 512, num_points)
        x = torch.cat((x1, x2, x3), dim=1).squeeze(-1)
        '''
        Get x.
        x: (batch_size, num_dims * 8, num_points, 1)
        num_dims = 64: (batch_size, 512, num_points, 1)
        '''

        if self.useBN:
            # (batch_size, self.emb_dims, num_points)
            x = self.act_f(self.bn3_lpd(self.conv3_lpd(x)))
        else:
            # (batch_size, self.emb_dims, num_points)
            x = self.act_f(self.conv3_lpd(x))
        # (batch_size, self.emb_dims, num_points, 1)
        x = x.unsqueeze(-1)
        '''
        Output: (batch_size, self.emb_dims, num_points, 1)
        '''
        return x


class LPD(nn.Module):
    def __init__(self, args):
        super(LPD, self).__init__()
        self.cycle = args.cycle
        self.negative_slope = 1e-2
        self.emb_dims = args.emb_dims
        self.num_points = args.num_points
        self.emb_nn = LPDNet(negative_slope=self.negative_slope)

    def forward(self, src, tgt):
        # (batch_size, num_dims, num_points)
        src = src
        tgt = tgt
        batch_size = src.size(0)
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        loss = self.getLoss(src, src_embedding, tgt_embedding)
        mse_ab_ = torch.mean((src_embedding - tgt_embedding) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab_ = torch.mean(torch.abs(src_embedding - tgt_embedding), dim=[0, 1, 2]).item() * batch_size

        return src_embedding, tgt_embedding, loss, mse_ab_, mae_ab_

    def getLoss(self, src, src_embedding, tgt_embedding):
        batch_size, _, num_points = src.size()
        # 取k个点对做实验
        k = 32
        nk = 8
        src = src[:, :, :k]
        src_embedding_k = src_embedding[:, :, :k]
        tgt_embedding_k = tgt_embedding[:, :, :k]
        _, num_dims, _ = tgt_embedding_k.size()
        nn.TripletMarginLoss()
        # 找到相距较远的点
        inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), src)  # [b,num,num]
        xx = torch.sum(src ** 2, dim=1, keepdim=True)  # [b,1,num] #x ** 2 表示点平方而不是x*x

        pairwise_distance = xx + inner
        pairwise_distance = pairwise_distance + xx.transpose(2, 1).contiguous()  # [b,num,num]

        # 每k个找到nk个最远的
        idx = pairwise_distance.topk(k=nk, dim=-1)[1]  # (batch_size, k, nk)
        # 获得索引阶梯数组
        idx_base = torch.arange(0, batch_size, device=torch.device('cuda')).view(-1, 1,
                                                                                 1) * k  # (batch_size, 1, 1) [0 k ... k*(B-1)]
        # 以batch为单位，加到索引上
        idx = idx + idx_base  # (batch_size, k, nk)
        # 展成一维数组，方便后续索引
        idx = idx.view(-1)  # (batch_size * k * nk)
        # 改变x的shape，方便索引。被索引数组是所有batch的所有点的特征，索引数组idx为所有临近点对应的序号，从而索引出所有领域点的特征
        # 取出目标点云对应的最远nk个点的特征
        topFarTgt = tgt_embedding_k.transpose(2, 1).contiguous().view(batch_size * k, -1)[idx,
                    :]  # (batch_size * k * nk,num_dims)
        # 统一数组形式
        src_embedding_shaped = src_embedding_k.transpose(2, 1).contiguous().view(batch_size, k, 1, num_dims).repeat(
            (1, 1, nk, 1)).view(batch_size * k * nk, -1)
        tgt_embedding_shaped = tgt_embedding_k.transpose(2, 1).contiguous().view(batch_size, k, 1, num_dims).repeat(
            (1, 1, nk, 1)).view(batch_size * k * nk, -1)

        triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
        loss_triplet = triplet_loss(src_embedding_shaped, tgt_embedding_shaped, topFarTgt)

        # 训练出的模长为1
        src_embedding = src_embedding.transpose(2, 1).contiguous()
        tgt_embedding = tgt_embedding.transpose(2, 1).contiguous()
        src_length = torch.norm(src_embedding, dim=-1)
        tgt_length = torch.norm(tgt_embedding, dim=-1)
        identity = torch.empty((batch_size, num_points), device=torch.device('cuda')).fill_(1)
        loss_norm1 = torch.sqrt(F.mse_loss(src_length, identity))
        loss_norm2 = torch.sqrt(F.mse_loss(tgt_length, identity))

        loss = loss_triplet + (loss_norm1 + loss_norm2) / 2.0 * 0.03

        return loss


