#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2020/9/5 10:10
# @FileName: build.py
# @Software: PyCharm


from .dcutils import loadData, loadPosition, trainDCnet
from mxnet import nd, gluon
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

class ObjectiveDCnet(object):

    def __init__(self, net_model,data_dir,relation_dir, **kwargs):
        self.net_model = net_model
        self.data_dir = data_dir #
        self.relation_dir = relation_dir
        self.trainer = kwargs.get('trainer', 'adam')
        self.lr = kwargs.get('lr', 1e-4)
        self.wd = kwargs.get('wd', 0)
        # self.activation = kwargs.get('activation', 'sigmoid')
        self.epoch = kwargs.get('epoch', 500)
        self.batch_size = kwargs.get('batch_size', 256)

    def prepareData(self):
        print('transfer data to ndarray')
        train_X, train_y, test_X, test_y, train_label, test_label = loadData(self.data_dir)
        X_1 = train_X.T #
        X_2 = test_X.T  #
        Y_1 = pd.concat([train_X.T, train_y.T], axis=1)  #train_y.T #
        Y_2 = pd.concat([test_X.T, test_y.T], axis=1)  #test_y.T  #
        print(f'train sample X_dim: {X_1.shape}, Y_dim: {Y_1.shape}')
        print(f'test  sample X_dim: {X_2.shape}, Y_dim: {Y_2.shape}')

        self.train_data = {'x': nd.array(X_1), 'y': nd.array(Y_1), 'label': train_label}
        self.test_data = {'x':  nd.array(X_2), 'y': nd.array(Y_2), 'label': test_label}

        self.input_ix = X_1.columns
        self.output_ix = Y_2.columns
        _, _, position = loadPosition(self.relation_dir)
        self.position = position
        self.output_dim = len(self.output_ix)
        print('ndarray data has generate.')

    def buildDCnet(self):
        print('build net model...')
        self.net = self.net_model(self.position, self.output_dim)
        self.net.initialize()

    def train(self):
        trainer = gluon.Trainer(self.net.collect_params(), self.trainer,
                                {'learning_rate': self.lr, 'wd': self.wd})
        loss = gluon.loss.L2Loss()
        error_epochs = trainDCnet(self.net, self.train_data, self.test_data,
                                  trainer, loss, epochs=self.epoch, batch_size=self.batch_size)
        return error_epochs

    def plotErrors(self, errors):
        X, Y, Z = [], [], []
        for epoch, train_error, test_error in errors:
            X.append(epoch)
            Y.append(train_error)
            Z.append(test_error)
        ln1, = plt.plot(X, Y, color='red')
        ln2, = plt.plot(X, Z, color='blue')
        plt.legend(handles=[ln1, ln2], labels=['train_loss', 'test_loss'])
        plt.show()

    def saveNet(self, save, extra=None):
        if extra is None:
            extra = ''
        print('save net...')
        save_data = {'net': self.net, 'input_ix': self.input_ix, 'output_ix':self.output_ix}
        name = f'net_M{self.position.shape[1]}_{self.trainer}_T{self.epoch}_{extra}.pkl'
        with open(os.path.join(save, name), 'wb') as f:
            pickle.dump(save_data, f)
        print(f'save success: {os.path.join(save, name)}')

    def autoRun(self):
        self.prepareData()
        self.buildDCnet()
        errors = self.train()
        self.plotErrors(errors)
        save_dir = os.path.join(os.getcwd(), 'DCNet_output')
        self.saveNet(save_dir)


