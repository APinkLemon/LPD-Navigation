# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年04月04日
"""

import os
import sys
import pickle
import numpy as np

with open("trainQueriesBaseline.pickle", 'rb') as handle:
    TRAINING_QUERIES = pickle.load(handle)
    print("Queries Loaded.")

print("start load fast")
DIR = "./"
TRAINING_POINT_CLOUD = []
fileName = "TRAINING_POINT_CLOUD_NEW.npy"
path = DIR + fileName

if os.path.exists(path):
    filename = TRAINING_QUERIES[0]["query"]
    print(filename)
    TRAINING_POINT_CLOUD = np.load(path)
    print("Fast Npy Loaded!")
else:
    for i in range(len(TRAINING_QUERIES)):
        filename = TRAINING_QUERIES[i]["query"]
        print(filename)
        pc = np.load("../" + filename)
        TRAINING_POINT_CLOUD.append(pc)
    TRAINING_POINT_CLOUD = np.asarray(TRAINING_POINT_CLOUD).reshape(-1, 4096, 3)
    np.save(path, TRAINING_POINT_CLOUD)
    print("Save Fast Npy: " + path)
