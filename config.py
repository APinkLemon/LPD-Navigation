# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月26日
"""

import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.path = edict()
__C.path.base = os.path.dirname(os.path.abspath(__file__))
