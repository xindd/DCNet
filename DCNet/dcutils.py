#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2020/9/5 10:10
# @FileName: dcutils.py
# @Software: PyCharm
import os
import pickle
import random
import tqdm
import matplotlib.pyplot as plt
from mxnet import autograd, nd
import mxnet as mx
import numpy as np


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

#迭代器
def dataIter(data, batch_size):
    features = data['x']
    output_y = data['y']
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  #
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), output_y.take(j)

# 加载数据
def loadData(dir):
    with open(dir, 'rb') as f:
        save_data = pickle.load(f)
    return save_data['train_X'], save_data['train_y'], \
           save_data['test_X'], save_data['test_y'], \
           save_data['train_label'], save_data['test_label']

# 加载位置矩阵
def loadPosition(dir):
    with open(dir, 'rb') as f:
        save_data = pickle.load(f)
    return save_data['marker_gene_exp'], save_data['non_marker_gene_exp'], save_data['position_mat']

# 评估测试误差
def evaluateAccuracy(net, data_iter, loss):
    error, num = 0.0, 0
    for X,y in data_iter:
        y_hat = net(X)
        l = loss(y.flatten(), y_hat.flatten()).sum()
        error += l.sum().asscalar()
        num += X.shape[0]
    return error / num

# 训练DCnet
def trainDCnet(net, train_data, test_data, trainer,  loss, epochs=5, batch_size=32):
    train_log = []
    for epoch in tqdm.tqdm(range(epochs), desc='train epochs'):
        train_l_sum, n = 0.0, 0
        for X,y in dataIter(train_data, batch_size):
            with autograd.record():
                y_hat = net(X)
                l = loss(y.flatten(), y_hat.flatten()).sum()
                # l = net(X, y).sum()
            l.backward()
            # print(net.weight_first.grad()[0,0:10])
            trainer.step(X.shape[0])

            n += X.shape[0]
            train_l_sum += l.asscalar()
        test_error = evaluateAccuracy(net, dataIter(test_data, batch_size), loss)
        train_log.append([epoch, train_l_sum/n, test_error])
        # train_log.append([epoch, train_l_sum/n, train_log])
    return train_log

# 微调
def fineTrain(net, train_data, trainer1, trainer2, loss, epochs=5):
    train_log = []
    net.collect_params().reset_ctx(mx.cpu())
    # net.hybridize()
    X, y, y_na = train_data
    y_mask = nd.array((~y_na).to_list()*y.shape[0]).reshape(y.shape[0],-1)
    X, y = X.as_in_context(mx.cpu()), y.as_in_context(mx.cpu())
    for epoch in tqdm.tqdm(range(epochs), desc='fine train epochs'):
        # X.attach_grad()
        with autograd.record():
            y_hat = net(X)
            l = loss(y_mask*y, y_mask*y_hat).sum()

        l.backward()
        # print(net.weight_first.grad().sum(axis=1))
        trainer1.step(X.shape[0])
        trainer2.step(X.shape[0])
        train_log.append([epoch, l.asscalar()])
    return train_log

def splitProfile(profile, index, fill_value=0):
    x = profile.reindex(index)
    x_na = np.isnan(x).sum(axis=1) >= 1
    x = x.fillna(fill_value)
    return x, x_na

