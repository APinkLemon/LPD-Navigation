# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年04月12日
"""

import cv2
import os
import sys
from dataProcess import getFilePathList


fileList = getFilePathList("Img")

# 读取时序图中的第一张图片
img = cv2.imread(fileList[0])

# 设置每秒读取多少张图片
fps = 2
imgInfo = img.shape

# 获取图片宽高度信息
size = (imgInfo[1], imgInfo[0])
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 定义写入图片的策略
videoWrite = cv2.VideoWriter('FirstEvaluateOther.mp4', fourcc, fps, size)

out_num = len(fileList)
for i in range(0, out_num):
    img = cv2.imread(fileList[i])
    print(i)
    # 将图片写入所创建的视频对象
    videoWrite.write(img)

videoWrite.release()
print('finish')
