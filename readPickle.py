# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年04月04日
"""

import pickle
import numpy as np
import glob

a = open("oxford_evaluation_database.pickle", "rb")
a = pickle.load(a)
b = open("GenerateDataBase/webots_evaluation_database.pickle", "rb")
seq_list = sorted(glob.glob("dataEvaluate2" + "/*"))
print(len(seq_list))
b = pickle.load(b)
print(b[1])
for i in range(58):
    print(b[1][i]['query'], seq_list[i])
    c = np.load(b[1][i]['query'])
print(len(b[1]))
seq_list = sorted(glob.glob("dataEvaluate2" + "/*"))
print(len(seq_list))
