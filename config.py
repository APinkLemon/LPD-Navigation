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
__C.path.query = "OxfordDataBase/"
__C.path.data = "OxfordDataBase/"
__C.path.pretrain = "Pretrain/model.pth"
__C.path.savePath = "Model"
__C.path.saveFile = "model1"
__C.path.logDir = "trainWriter"

__C.train = edict()
__C.train.device = "cuda:0"
__C.train.cudnn = True
__C.train.parallel = False
__C.train.loadFast = True
__C.train.lr = 0.001
__C.train.momentum = 0.9
__C.train.batchQueries = 1
__C.train.batchEval = 1

__C.train.numPoints = 4096
__C.train.embDims = 1024
__C.train.featureNet = "lpdnetorigin"
__C.train.xyzTransform = False
__C.train.featureTransform = False

__C.train.maxEpoch = 10
__C.train.lossFunction = "quadruplet"
# __C.train.lossFunction = "triplet"
__C.train.optimizer = "momentum"
# __C.train.optimizer = "adam"

__C.train.positives_per_query = 1
__C.train.negatives_per_query = 2
__C.train.hard_neg_per_query = 2

__C.loss = edict()
__C.loss.margin1 = 0.5
__C.loss.margin2 = 0.2
__C.loss.triplet_use_best_positives = True
__C.loss.loss_lazy = True
__C.loss.ignore_zero_loss = False


__C.net = edict()
__C.net.cat_or_stack = True
