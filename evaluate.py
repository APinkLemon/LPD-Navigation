# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月29日
"""

import sys
from dataLoader import *
from torch.backends import cudnn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from dataProcess import *
import math


print('#' * 40)
print("This is evaluate! ")
cudnn.enabled = True
recall_num = 10
EVAL_DATABASE_FILE = 'GenerateDataBase/webots_evaluation_database.pickle'
EVAL_QUERY_FILE = 'GenerateDataBase/webots_evaluation_query.pickle'
DATABASE_SETS = get_sets_dict(EVAL_DATABASE_FILE)
QUERY_SETS = get_sets_dict(EVAL_QUERY_FILE)
print('#' * 40)


def evaluate_model(model, tqdm_flag=True):
    # 计算 Recall @N
    recall = np.zeros(recall_num)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    torch.cuda.empty_cache()
    if tqdm_flag:
        fun_tqdm = tqdm
    else:
        fun_tqdm = list

    # 总共23个子图
    # 获得每个子地图的每一帧点云的描述子
    for i in fun_tqdm(range(len(DATABASE_SETS))):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))

    # 获得每个子地图的每一帧要被评估的点云的描述子
    for j in fun_tqdm(range(len(QUERY_SETS))):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

    torch.cuda.empty_cache()
    for m in fun_tqdm(range(len(QUERY_SETS))):
        for n in range(len(QUERY_SETS)):
            if m == n:
                continue
            # 寻找当前第m个子图和第n个子图的检索结果
            pair_recall, pair_similarity, pair_opr = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    # 不求均值就可以得到@N的recall
    ave_recall = np.mean(np.mean(recall / count))
    if tqdm_flag:
        print("ave_recall: "+str(ave_recall))
    else:
        print("ave_recall: "+str(ave_recall), print_flag=False)

    # print(similarity)
    average_similarity_score = np.mean(similarity)
    if tqdm_flag:
        print("average_similarity_score: "+str(average_similarity_score))
    else:
        print("average_similarity_score: "+str(average_similarity_score), print_flag=False)
    #
    ave_one_percent_recall = np.mean(one_percent_recall)
    if tqdm_flag:
        print("ave_one_percent_recall: "+str(ave_one_percent_recall))
    else:
        print("ave_one_percent_recall: "+str(ave_one_percent_recall), print_flag=False)

    return ave_recall, average_similarity_score, ave_one_percent_recall


def get_latent_vectors(model, dict_to_process, device=cfg.train.device):
    model.eval()
    torch.cuda.empty_cache()
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.train.batchEval * \
        (1 + cfg.train.positives_per_query + cfg.train.negatives_per_query)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*batch_num]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)
        # print("load time: ", time() - start)
        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            # print(feed_tensor.mean(dim=[0, 1, 2]))
            out = model(feed_tensor)
        # print("forward time: ", time() - start)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)
        # del feed_tensor
        # out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if len(q_output) != 0:
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            output = model(feed_tensor)
        # del feed_tensor
        output = output.detach().cpu().numpy()
        output = np.squeeze(output)
        if q_output.shape[0] != 0:
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    torch.cuda.empty_cache()
    model.train()
    # print(q_output.shape)
    # print(q_output.shape, np.asarray(q_output).mean(),np.asarray(q_output).reshape(-1,256).min(),
    # np.asarray(q_output).reshape(-1,256).max())
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, render = True):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    if render:
        print(database_output.shape)
        print(queries_output.shape)
    xtList = []
    xpList = []
    ytList = []
    ypList = []
    barX = [1 + i for i in range(len(queries_output))]
    barY = [0 for _ in range(len(queries_output))]

    database_nbrs = KDTree(database_output)

    recall = [0] * recall_num
    top1_similarity_score = []
    one_percent_retrieved = 0
    # threshold之内对应百分之一
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    # 遍历需要被评估的点云
    for i in range(len(queries_output)):
        # 得到该点云在第m个子图的真实检索点云序列
        true_neighbors = QUERY_SETS[n][i][m]
        if len(true_neighbors) == 0:
            continue
        # 被评估数
        num_evaluated += 1
        # 得到该点云在第m个子图的实际检索点云序列
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=recall_num)
        if render:
            print("%" * 150)
            print("This is True: ")
            print(i)
            print(QUERY_SETS[n][i])
            xT = QUERY_SETS[n][i]['northing']
            yT = QUERY_SETS[n][i]['easting']
            print("This is output: ")
            print(true_neighbors)
            print(indices[0][0])
            print(DATABASE_SETS[m][indices[0][0]])
            xP = DATABASE_SETS[m][indices[0][0]]['northing']
            yP = DATABASE_SETS[m][indices[0][0]]['easting']

            gs = GridSpec(3, 12)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[0:2, 0:3])
            ax1.scatter(xtList, ytList, marker='x', color='b')
            ax1.scatter(xpList, ypList, marker='x', color='r')
            ax1.scatter([xT], [yT], marker='*', color='b')
            ax1.scatter([xP], [yP], marker='*', color='r')
            xtList.append(xT)
            xpList.append(xP)
            ytList.append(yT)
            ypList.append(yP)
            plt.xlim((-60, 120))
            plt.ylim((-50, 120))

            dataTrue = np.load(QUERY_SETS[n][i]['query'])
            x1 = dataTrue[:, 0]
            y1 = dataTrue[:, 1]
            z1 = dataTrue[:, 2]

            dataEval = np.load(DATABASE_SETS[m][indices[0][0]]['query'])
            x2 = dataEval[:, 0]
            y2 = dataEval[:, 1]
            z2 = dataEval[:, 2]

            ax2 = fig.add_subplot(gs[0:2, 4:7], projection='3d')
            ax2.scatter(x1, y1, z1, s=1)
            plt.xlim((-0.5, 0.5))
            plt.ylim((-0.5, 0.5))

            ax2.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
            ax2.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
            ax2.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
            ax2.view_init(elev=60, azim=45)

            ax3 = fig.add_subplot(gs[0:2, 8:11], projection='3d')
            ax3.scatter(x2, y2, z2, s=1)
            plt.xlim((-0.5, 0.5))
            plt.ylim((-0.5, 0.5))

            ax3.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
            ax3.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
            ax3.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
            ax3.view_init(elev=60, azim=45)

            barY[i] = ((xT - xP) ** 2 + (yT - yP) ** 2) ** 0.5
            print(barY[i])

            ax4 = fig.add_subplot(gs[2, 0:12])
            ax4.bar(barX, barY)
            plt.ylim((0, 200))

            plt.tight_layout()
            plt.show()

        # 遍历recal_num得到在不同指标下的结果
        for j in range(len(indices[0])):
            # 如果第j个候选是真实值
            if indices[0][j] in true_neighbors:
                # 如果是top1 recall
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                # 如果第j个候选是真实值，+1
                recall[j] += 1
                break
        # 如果前百分之一包含真实值，+1
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    # 这里用cumsum因为第j个元素只代表第j个，不代表前j个
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall
