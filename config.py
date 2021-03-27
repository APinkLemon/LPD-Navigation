# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月26日
"""

from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.path = edict()
__C.path.raw = "dataSet1/"
__C.path.query = "GenerateDataBase/"
__C.path.data = "GenerateDataBase/"
__C.path.pretrain = "Pretrain/"

__C.train = edict()
__C.train.lossFunction = "quadruplet"
# __C.train.lossFunction = "triplet"
__C.train.optimizer = "momentum"
# __C.train.optimizer = "adam"
__C.train.lr = 0.001
__C.train.momentum = 0.9
__C.train.numPoints = 4096
__C.train.embDims = 1024
__C.train.featureNet = "lpdnetorigin"
__C.train.xyzTransform = False
__C.train.featureTransform = False

__C.net = edict()
__C.net.cat_or_stack = True
