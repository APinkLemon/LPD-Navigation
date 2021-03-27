# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月27日
"""

import os
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from config import cfg
import pointnetVlad as PNV
import pointnetVladLoss as PNV_loss
from dataLoader import Oxford_train_base, Oxford_train_advance


def train():
    device = cfg.train.device
    model = PNV.PointNetVlad(
        emb_dims=cfg.train.embDims,
        num_points=cfg.train.numPoints,
        featnet=cfg.train.featureNet,
        xyz_trans=cfg.train.xyzTransform,
        feature_transform=cfg.train.featureTransform
    )

    if torch.cuda.is_available():
        model = model.cuda(device)
    else:
        model = model.cpu()

    if not os.path.exists(cfg.path.pretrain):
        print("Not Find PreTrained Network! ")
    else:
        model.load_state_dict(torch.load(cfg.path.pretrain), strict=False)
        print("Load PreTrained Network! ")

    if cfg.train.parallel:
        if torch.cuda.device_count() > 1:
            model = nn.parallel.DataParallel(model)
            print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
        else:
            print("Let's use " + device)
    else:
        print("Let's use " + device)

    if cfg.train.lossFunction == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper

    if cfg.train.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), cfg.train.lr, momentum=cfg.train.momentum)
    elif cfg.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr)

    loader_base = DataLoader(Oxford_train_base(args=cfg.train), batch_size=cfg.train.batchQueries,
                             shuffle=False, drop_last=True, num_workers=4)
    loader_advance = DataLoader(Oxford_train_advance(args=cfg.train, model=model), batch_size=cfg.train.batchQueries,
                                shuffle=False, drop_last=True, num_workers=4)


if __name__ == "__main__":
    cudnn.enabled = cfg.train.cudnn
    train()
    print("Train Finished!")
