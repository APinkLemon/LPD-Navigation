# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年04月04日
"""

from config import cfg
from dataProcess import *
from dataTransform import *


base = cfg.path.raw
fileList = getFilePathList(base)
print(len(fileList))
for i in range(len(fileList)):
    print("#" * 150)
    print(i)
    file = pathToNpyPath(fileList[i])
    print(file)
    a = np.load(file)
    exp = npyToPointCloud(a)
    exp = rotatePointCloud(exp)
    exp = pointCloudToNpy(exp)
    b, c = RemoveGround(exp)
    exp = npyToPointCloud(b)
    newExp = downPcdVoxel(exp)
    exp = pointCloudToNpy(newExp)
    savePath = "dataEvaluate2" + pathToNpyPath(fileList[i])[11:]
    print(savePath)
    np.save(savePath, exp)
