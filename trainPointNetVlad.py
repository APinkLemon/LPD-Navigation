# -*- coding:utf-8 -*-
"""
作者：34995
日期：2021年03月27日
"""

import sys
import torch.nn as nn
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import evaluate
import pointnetVlad as PNV
import pointnetVladLoss as PNV_loss
from dataLoader import *


def save_model(model, epoch, optimizer, ave_one_percent_recall, best_ave_one_percent_recall, TOTAL_ITERATIONS):
    best_ave_one_percent_recall = best_ave_one_percent_recall
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    save_name = cfg.path.savePath + '/' + str(epoch) + "-" + cfg.path.saveFile
    torch.save({
        'epoch': epoch,
        'iter': TOTAL_ITERATIONS,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'recall': ave_one_percent_recall,
    }, save_name)
    print("Model Saved As " + save_name)

    if best_ave_one_percent_recall < ave_one_percent_recall:
        best_ave_one_percent_recall = ave_one_percent_recall
        save_name = cfg.path.savePath + '/' + "best" + "-" + cfg.path.saveFile
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'recall': ave_one_percent_recall,
        }, save_name)
        print("Model Saved As " + save_name)
    return best_ave_one_percent_recall


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    # print_gpu("2")
    feed_tensor = torch.cat((queries, positives, negatives, other_neg), 1)
    feed_tensor = feed_tensor.view((-1, 1, cfg.train.numPoints, 3))
    # feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.cuda()
    # print_gpu("3")
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(cfg.train.batchQueries, -1, cfg.train.featureDim)
    o1, o2, o3, o4 = torch.split(
        output, [1, cfg.train.positives_per_query, cfg.train.negatives_per_query, 1], dim=1)
    return o1, o2, o3, o4


def train():
    TOTAL_ITERATIONS = 0
    starting_epoch = 0
    division_epoch = 7
    best_ave_one_percent_recall = 0

    device = cfg.train.device
    model = PNV.PointNetVlad(
        emb_dims=cfg.train.embDims,
        num_points=cfg.train.numPoints,
        featnet=cfg.train.featureNet,
        xyz_trans=cfg.train.xyzTransform,
        feature_transform=cfg.train.featureTransform
    )

    if torch.cuda.is_available():
        model = model.cuda(device)
    else:
        model = model.cpu()

    if not os.path.exists(cfg.path.pretrain):
        print("Not Find PreTrained Network! ")
    else:
        model.load_state_dict(torch.load(cfg.path.pretrain), strict=False)
        print("Load PreTrained Network! ")

    if cfg.train.parallel:
        if torch.cuda.device_count() > 1:
            model = nn.parallel.DataParallel(model)
            print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
        else:
            print("Let's use " + device)
    else:
        print("Let's use " + device)

    if cfg.train.lossFunction == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper

    if cfg.train.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), cfg.train.lr, momentum=cfg.train.momentum)
    elif cfg.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr)
    else:
        print("Optimizer is wrong!")
        sys.exit(999)

    loader_base = DataLoader(Oxford_train_base(args=cfg.train), batch_size=cfg.train.batchQueries,
                             shuffle=False, drop_last=True, num_workers=4)
    loader_advance = DataLoader(Oxford_train_advance(args=cfg.train, model=model), batch_size=cfg.train.batchQueries,
                                shuffle=False, drop_last=True, num_workers=4)

    if starting_epoch > division_epoch + 1:
        update_vectors(model, device)

    train_writer = SummaryWriter(os.path.join(cfg.path.logDir, 'train_writer'))
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=2, verbose=True, threshold=0.1, min_lr=0.00001)

    for epoch in range(starting_epoch, cfg.train.maxEpoch):
        print('**** EPOCH %03d ****' % epoch)
        TOTAL_ITERATIONS = train_one_epoch(model, device, division_epoch, TOTAL_ITERATIONS, optimizer, train_writer,
                                           loss_function, epoch, loader_base, loader_advance)
        print("learn rate " + str(optimizer.param_groups[0]['lr']))
        print('EVALUATING...')
        cfg.path.outputFile = cfg.path.resultsFolder + 'results_' + str(epoch) + '.txt'
        ave_recall, average_similarity_score, ave_one_percent_recall = evaluate.evaluate_model(model, tqdm_flag=True)
        print('EVAL %% RECALL: %s' % str(ave_one_percent_recall))

        TOTAL_ITERATIONS = save_model(model, epoch, optimizer, ave_one_percent_recall,
                                      best_ave_one_percent_recall, TOTAL_ITERATIONS)
        scheduler.step(ave_one_percent_recall)
        train_writer.add_scalar("Val Recall", ave_one_percent_recall, epoch)


def train_one_epoch(model, device, division_epoch, TOTAL_ITERATIONS, optimizer, train_writer,
                    loss_function, epoch, loader_base, loader_advance):
    TOTAL_ITERATIONS = TOTAL_ITERATIONS
    batch_num = cfg.train.batchQueries
    if epoch <= division_epoch:
        for queries, positives, negatives, other_neg in tqdm(loader_base):
            model.train()
            optimizer.zero_grad()
            output_queries, output_positives, output_negatives, output_other_neg = \
                run_model(model, queries, positives, negatives, other_neg)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg,
                                 cfg.loss.margin_1, cfg.loss.margin_2, use_min=cfg.loss.triplet_use_best_positives,
                                 lazy=cfg.loss.loss_lazy, ignore_zero_loss=cfg.loss.ignore_zero_loss)
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("epoch", epoch, TOTAL_ITERATIONS)
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            train_writer.add_scalar("learn rate", optimizer.param_groups[0]['lr'], TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += batch_num

            # if (TOTAL_ITERATIONS % (6000 // batch_num * batch_num) == 0):
            #     log_string('EVALUATING...', print_flag=False)
            #     ave_recall, average_similarity_score, ave_one_percent_recall
            #     = evaluate.evaluate_model(para.model, tqdm_flag=False)
            #     log_string('EVAL %% RECALL: %s' % str(ave_one_percent_recall), print_flag=True)
            # train_writer.add_scalar("one percent recall", ave_one_percent_recall, TOTAL_ITERATIONS)

    else:
        if epoch == division_epoch + 1:
            update_vectors(model, device)
        for queries, positives, negatives, other_neg in tqdm(loader_advance):
            model.train()
            optimizer.zero_grad()
            output_queries, output_positives, output_negatives, output_other_neg = \
                run_model(model, queries, positives, negatives, other_neg)
            # log_string("train: ",time()-start)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg,
                                 cfg.loss.margin_1, cfg.loss.margin_2, use_min=cfg.loss.triplet_use_best_positives,
                                 lazy=cfg.loss.loss_lazy, ignore_zero_loss=cfg.loss.ignore_zero_loss)
            # log_string("train: ",time()-start)
            # 比较耗时
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("epoch", epoch, TOTAL_ITERATIONS)
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            train_writer.add_scalar("learn rate", optimizer.param_groups[0]['lr'], TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += cfg.train.batchQueries
            if TOTAL_ITERATIONS % (int(700 * (epoch + 1))//batch_num*batch_num) == 0:
                update_vectors(model, device, tqdm_flag=False)
            # if (TOTAL_ITERATIONS % (int(1000 * (epoch + 1)) // batch_num * batch_num) == 0):
            #     ave_recall, average_similarity_score, ave_one_percent_recall =
            #     evaluate.evaluate_model(para.model, tqdm_flag=False)
            #     log_string('EVAL %% RECALL: %s' % str(ave_one_percent_recall), print_flag=True)
            #     train_writer.add_scalar("one percent recall", ave_one_percent_recall, TOTAL_ITERATIONS)

    return TOTAL_ITERATIONS


if __name__ == "__main__":
    trainMode = 0
    cudnn.enabled = cfg.train.cudnn
    if trainMode:
        print("Start Train!")
        train()
        print("Train Finished!")
    else:
        print("Start Eval!")
        model = PNV.PointNetVlad(
            emb_dims=cfg.train.embDims,
            num_points=cfg.train.numPoints,
            featnet=cfg.train.featureNet,
            xyz_trans=cfg.train.xyzTransform,
            feature_transform=cfg.train.featureTransform
        )

        device = cfg.train.device
        if torch.cuda.is_available():
            model = model.cuda(device)
        else:
            model = model.cpu()

        if os.path.exists(cfg.path.pretrain):
            # print("*"*100)
            # for key, value in torch.load(cfg.path.pretrain)['state_dict'].items():
            #     print(key)
            # sys.exit(996)
            modelDict = torch.load(cfg.path.pretrain)
            model.load_state_dict(modelDict["state_dict"], strict=True)
            print("Load Pretrained Model!")
        else:
            sys.exit(995)

        # for name, parameters in model.named_parameters():
        #     print(name, ':')

        ave_recall, average_similarity_score, ave_one_percent_recall = evaluate.evaluate_model(model, tqdm_flag=True)

        print("ave_one_percent_recall:", ave_one_percent_recall)
