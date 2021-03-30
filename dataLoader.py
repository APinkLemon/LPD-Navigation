# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月27日
"""

import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from config import cfg
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree


def flat(li):
    for k in li:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def get_sets_dict(filename):
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Sets Loaded: " + filename)
        return trajectories


def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded: " + filename)
        return queries


def load_pc_file(filename):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join(cfg.DATASET_FOLDER, filename), dtype=np.float64)

    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc


def load_pc_files(filenames):
    pcs = []
    for filename in filenames:
        # log_string(filename)
        pc = load_pc_file(filename)
        if(pc.shape[0] != 4096):
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def update_vectors(model, device, tqdm_flag=True):
    global TRAINING_LATENT_VECTORS

    torch.cuda.empty_cache()

    if tqdm_flag:
        fun_tqdm = tqdm
    else:
        fun_tqdm = list

    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))

    batch_num = cfg.train.batchEval * (1 + cfg.train.positives_per_query + cfg.train.negatives_per_query)
    # log_string("\n args: ",args.batch_num_queries,args.positives_per_query,args.negatives_per_query)
    q_output = []

    model.eval()

    for q_index in fun_tqdm(range(len(train_file_idxs) // batch_num)):
    # for q_index in tqdm(range(batch_num*2 // batch_num)):
        if load_fast:
            file_indices = np.arange(q_index * batch_num, (q_index + 1) * (batch_num))
            queries = TRAINING_POINT_CLOUD[file_indices]
        else:
            file_indices = train_file_idxs[q_index * batch_num:(q_index + 1) * (batch_num)]
            file_names = []
            for index in file_indices:
                file_names.append(TRAINING_QUERIES[index]["query"])
            queries = load_pc_files(file_names)

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)
        q_output.append(out)

    q_output = np.array(q_output)
    if (len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    del feed_tensor

    # handle edge case
    for q_index in fun_tqdm(range((len(train_file_idxs) // batch_num * batch_num), len(TRAINING_QUERIES.keys()))):
        if load_fast:
            queries = TRAINING_POINT_CLOUD[train_file_idxs[q_index]]
            queries = np.expand_dims(np.expand_dims(queries, axis=0), axis=0)
        else:
            index = train_file_idxs[q_index]
            queries = load_pc_files([TRAINING_QUERIES[index]["query"]])

        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            queries_tensor = queries_tensor.to(device)
            output = model(queries_tensor)

        output = output.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output.reshape(-1,cfg.FEATURE_OUTPUT_DIM)))
        else:
            q_output = output
    del queries_tensor

    model.train()

    TRAINING_LATENT_VECTORS = q_output
    # log_string("Updated cached feature vectors")
    torch.cuda.empty_cache()

    if tqdm_flag:
        print("update all vectors.")
    else:
        print("update all vectors.", print_flag=False)


def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    # get query tuple for dictionary entry
    # return list [query,positives,negatives]
    query = load_pc_file(dict_value["query"])  # Nx3
    random.shuffle(dict_value["positives"])
    pos_files = []

    # 不必考虑正样本是否充足，因为之前判断过
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    positives = load_pc_files(pos_files)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        # 如果hard不够，再进行补充
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files)

    # log_string("load time: ",time()-start)
    # 是否需要额外的neg（Quadruplet loss需要）
    if other_neg is False:
        return [query, positives, negatives]
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        # 减去与neighbors公共有的部分，剩下既不进也不远的那些部分
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)
        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]
        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"]) # 就一个

        return [query, positives, negatives, neg2]


def get_query_tuple_fast(item, dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    # get query tuple for dictionary entry
    # return list [query,positives,negatives]
    query = TRAINING_POINT_CLOUD[item] # 就一个点云
    random.shuffle(dict_value["positives"])
    # 不必考虑正样本是否充足，因为之前判断过
    positives = TRAINING_POINT_CLOUD[(dict_value["positives"][:num_pos])]

    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        neg_indices=dict_value["negatives"][:num_neg]
    else:
        neg_indices.append(hard_neg)
        j = 0
        # 如果hard不够，再进行补充
        neg_indices = list(flat(neg_indices))
        while(len(neg_indices) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    neg_indices = list(flat(neg_indices))
    negatives = TRAINING_POINT_CLOUD[neg_indices]

    # log_string("load time: ",time()-start)
    # 是否需要额外的neg（Quadruplet loss需要）
    if other_neg is False:
        return [query, positives, negatives]
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        # 减去与neighbors公共有的部分，剩下既不进也不远的那些部分
        neighbors = list(flat(neighbors))
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)
        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]
        neg2 = TRAINING_POINT_CLOUD[possible_negs[0]] # 就一个

        return [query, positives, negatives, neg2]


def get_random_hard_negatives(query_vec, random_negs, hard_neg_num):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=hard_neg_num)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs


def get_feature_representation(filename, model, device):
    model.eval()
    queries = load_pc_files([filename])
    queries = np.expand_dims(queries, axis=1)
    # if(BATCH_NUM_QUERIES-1>0):
    #    fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
    #    q=np.vstack((queries,fake_queries))
    # else:
    #    q=queries
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output


class Oxford_train_base(Dataset):
    def __init__(self, args):
        self.num_points = args.numPoints
        self.positives_per_query = args.positives_per_query
        self.negatives_per_query = args.negatives_per_query
        self.train_len = len(TRAINING_QUERIES.keys())
        print('Load Oxford Dataset')
        self.last = []

    def __getitem__(self, item):
        if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
            if self.last == []:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]

        # no cached feature vectors
        if load_fast:
            q_tuples = get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query,
                                            self.negatives_per_query,
                                            TRAINING_QUERIES, hard_neg=[], other_neg=True)
        else:
            q_tuples = get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                       TRAINING_QUERIES, hard_neg=[], other_neg=True)

        # 对点云进行增强，旋转或者加噪声
        # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[item],
        # POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
        # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[item],
        # POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

        # 这里默认使用了quadruplet loss，所以必须找到other_neg
        if (q_tuples[3].shape[0] != self.num_points):
            print('----' + 'FAULTY other_neg' + '-----')
            if self.last == []:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]

        queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=0)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=0)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(queries.shape) != 3):
            print('----' + 'FAULTY QUERY' + '-----')
            if self.last == []:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]
        self.last = [queries, positives, negatives, other_neg]
        return queries.astype('float32'), positives.astype('float32'), negatives.astype('float32'), other_neg.astype(
            'float32')

    def __len__(self):
        return self.train_len


class Oxford_train_advance(Dataset):
    def __init__(self, args, model):
        self.model = model
        self.num_points = args.numPoints
        self.positives_per_query = args.positives_per_query
        self.negatives_per_query = args.negatives_per_query
        self.train_len = len(TRAINING_QUERIES.keys())
        print('Load Oxford Dataset')
        print(self.train_len)

        self.sampled_neg = 4000
        self.hard_neg_num = args.hard_neg_per_query
        if self.hard_neg_num > args.negatives_per_query:
            print("self.hard_neg_num >  args.negatives_per_query")
        self.last = []

    def __getitem__(self, item):
        if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
            if self.last == []:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]

        if (len(HARD_NEGATIVES.keys()) == 0):
            from time import time
            start = time()
            query = get_feature_representation(TRAINING_QUERIES[item]['query'], self.model)
            random.shuffle(TRAINING_QUERIES[item]['negatives'])
            negatives = TRAINING_QUERIES[item]['negatives'][0:self.sampled_neg]
            hard_negs = get_random_hard_negatives(query, negatives, self.hard_neg_num)
            if load_fast:
                q_tuples = get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query,
                                                self.negatives_per_query,
                                                TRAINING_QUERIES, hard_neg=hard_negs, other_neg=True)
            else:
                q_tuples = get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                           TRAINING_QUERIES, hard_neg=hard_negs, other_neg=True)

        #     如果指定了一些HARD_NEGATIVES，實際沒有
        else:
            query = get_feature_representation(
                TRAINING_QUERIES[item]['query'], self.model)
            random.shuffle(TRAINING_QUERIES[item]['negatives'])
            negatives = TRAINING_QUERIES[item
                        ]['negatives'][0:self.sampled_neg]
            hard_negs = get_random_hard_negatives(
                query, negatives, self.hard_neg_num)
            hard_negs = list(set().union(
                HARD_NEGATIVES[item], hard_negs))
            # print('hard', hard_negs)
            if load_fast:
                q_tuples = get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query,
                                                self.negatives_per_query,
                                                TRAINING_QUERIES, hard_neg=hard_negs, other_neg=True)
            else:
                q_tuples = get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                           TRAINING_QUERIES, hard_negs, other_neg=True)

        # 这里默认使用了quadruplet loss，所以必须找到other_neg
        if (q_tuples[3].shape[0] != self.num_points):
            print('----' + 'FAULTY other_neg' + '-----')
            if self.last == []:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]

        queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=0)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=0)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(queries.shape) != 3):
            print('----' + 'FAULTY QUERY' + '-----')
            if self.last == []:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]
        self.last = [queries, positives, negatives, other_neg]
        return queries.astype('float32'), positives.astype('float32'), negatives.astype('float32'), other_neg.astype('float32')

    def __len__(self):
        return self.train_len


print('#' * 40)
print("This is dataProcess!")
load_fast = cfg.train.loadFast
HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []
TEST_QUERIES = get_queries_dict(cfg.path.query + "test_queries_baseline.pickle")
TRAINING_QUERIES = get_queries_dict(cfg.path.query + "training_queries_baseline.pickle")
TRAINING_POINT_CLOUD = np.load(cfg.path.data + "TRAINING_POINT_CLOUD.npy")
print('#' * 40)
