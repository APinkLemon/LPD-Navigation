# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月26日
"""

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from config import cfg


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if point[0] - x_width < northing < point[0] + x_width and point[1] - y_width < easting < point[1] + y_width:
            in_test_set = True
            break
    return in_test_set


def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(folders, p, output_name):
    database_trees = []
    test_trees = []
    for folder in folders:
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(folder, sep=',')
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], p, x_width, y_width):
                df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in folders:
        print(folder)
        database = {}
        test = {}
        df_locations = pd.read_csv(folder, sep=',')
        df_locations['timestamp'] = folder[3:-4] + "/" + df_locations['timestamp'].astype(str) + ".npy"
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], p, x_width, y_width):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'],
                                          'easting': row['easting']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array(
                    [[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=25)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, output_name + '_evaluation_query.pickle')


x_width = 10
y_width = 10

p1 = [5735712.768124, 620084.402381]
p2 = [5735611.299219, 620540.270327]
p3 = [5735237.358209, 620543.094379]
p4 = [5734749.303802, 619932.693364]

p5 = [-45.63004927531634, 52.89379211990181]
p6 = [104.21996920847565, 59.481325480907614]
p7 = [93.33673491848327, 95.09738907151828]
p8 = [67.37772388791976, -44.74183118168341]

p_dict = {"oxford": [p1, p2, p3, p4], "webots": [p5, p6, p7, p8]}
folders_in = ["../dataEvaluate1.csv", "../dataEvaluate2.csv"]
p_webots = p_dict["webots"]
construct_query_and_database_sets(folders_in, p_webots, "webots")
