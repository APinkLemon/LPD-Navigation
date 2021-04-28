# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月14日
"""

# import open3d as o3d
import numpy as np
import glob
import os


def storeDataToNumpy(path, render=True):
    j = 0
    data = []
    for line in open(path):
        single_data = line.split(' ')[0:-2]
        if j == 0:
            j += 1
            continue
        for i in range(len(single_data)):
            single_data[i] = float(single_data[i])
        if abs(single_data[0]) > 10000 or abs(single_data[1]) > 10000 or abs(single_data[2]) > 10000:
            continue
        if render:
            print(single_data)
        data.append(single_data)

    data_numpy = np.array(data)
    savePath = pathToNpyPath(path)
    if os.path.exists(savePath):
        print(path + " Exists!")
        return savePath
    np.save(savePath, data_numpy)
    print(data_numpy.shape)
    print("Numpy " + savePath + " Successfully Saved!")
    return savePath


def storeDataToPcd(path):
    data_numpy = np.load(path).reshape(-1, 3)
    print(data_numpy.shape)
    exp = o3d.geometry.PointCloud()
    exp.points = o3d.utility.Vector3dVector(data_numpy)
    savePath = pathToPcdPath(path)
    if os.path.exists(savePath):
        print(path + " Exists!")
        return savePath
    o3d.io.write_point_cloud(savePath, exp)
    print("Pcd " + savePath + " Successfully Saved!")
    return savePath


def visionPointCloud(pointCloud):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0.0, 0.0, 0.0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pointCloud)
    vis.add_geometry(axis)
    vis.update_geometry(pointCloud)
    vis.update_geometry(axis)
    vis.poll_events()
    vis.update_renderer()
    vis.run()


def visionPointCloudPcd(path):
    exp = o3d.io.read_point_cloud(path)
    visionPointCloud(exp)


def getFilePathList(basePath):
    seq_list = sorted(glob.glob(basePath + "/*"))
    print(seq_list)
    return seq_list


def pathToBaseNpyPath(basePath):
    return basePath[:-2] + "Npy" + basePath[-2] + "/"


def pathToBasePcdPath(basePath):
    return basePath[:-2] + "Pcd" + basePath[-2] + "/"


def pathToNpyPath(path):
    return path.replace("Set", "SetNpy").replace(".txt", ".npy")


def pathToPcdPath(path):
    if "SetNpy" in path:
        return path.replace("SetNpy", "SetPcd").replace(".npy", ".pcd")
    elif "Set" in path:
        return path.replace("Set", "SetPcd").replace(".txt", ".pcd")
    else:
        return 9999


def pointCloudToNpy(pointCloud):
    return np.asarray(pointCloud.points)


def npyToPointCloud(npy):
    pointCloud = o3d.geometry.PointCloud()
    pointCloud.points = o3d.utility.Vector3dVector(npy)
    return pointCloud


# def storeDataToOne(path):
#     if os.path.exists(path + "TRAINING_POINT_CLOUD.npy"):
#         TRAINING_POINT_CLOUD = np.load(path + "TRAINING_POINT_CLOUD.npy")
#     else:

