# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月26日
"""

from config import cfg
import pandas as pd
from dataProcess import getFilePathList


pathList = getFilePathList("../dataSet1/")
posX = []
posY = []
filename = []
for i in pathList:
    print(i[-14:-4])
    filename.append(i[-14:-4])
    for line in open(i):
        single_data = line[1:-2].split(', ')
        single_data = list(map(float, single_data))
        posX.append(single_data[0])
        posY.append(single_data[2])
        break
dataDict = {"filename": filename, "posX": posX, "posY": posY}
data = pd.DataFrame(dataDict)
print(data)
data.to_csv(
    '../dataTrain1.csv',
    index=False,  # 不保存行索引
    header=True,  # 保存列索引
)
