# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月27日
"""

import torch
from config import cfg
import pointnetVlad as PNV
import pointnetVladLoss as PNV_loss


def train():
    model = PNV.PointNetVlad(
        emb_dims=cfg.train.embDims,
        num_points=cfg.train.numPoints,
        featnet=cfg.train.featureNet,
        xyz_trans=cfg.train.xyzTransform,
        feature_transform=cfg.train.featureTransform
    )
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    if cfg.train.lossFunction == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper

    if cfg.train.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), cfg.train.lr, momentum=cfg.train.momentum)
    elif cfg.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr)


if __name__ == "__main__":
    train()
    print("Train Finished!")
