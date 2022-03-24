#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2020/9/5 10:09
# @FileName: models.py
# @Software: PyCharm


from mxnet import gluon, nd
from mxnet import init

class MyInit(init.Initializer):
    def __init__(self, relation):
        super(MyInit, self).__init__()
        self.relation = relation
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.array(self.relation)*0.01

class DCNet(gluon.nn.Block):
    def __init__(self, position_mat, output_units):
        super().__init__()
        self.position = nd.array(position_mat)
        input_units, middle_units = position_mat.shape

        self.weight_first = self.params.get('weight_first', shape=(input_units, middle_units), init=MyInit(relation=self.position))
        self.second_layer = gluon.nn.Dense(output_units, activation='relu')

    def forward(self, x):
        self.singure = self.weight_first.data() * self.position
        self.first_layer = nd.Activation(nd.dot(x, self.singure), act_type='relu')
        self.output = self.second_layer(self.first_layer)
        return self.output

class DCNetNoise(gluon.nn.Block):
    def __init__(self, position_mat, output_units, dropout=0.1):
        super().__init__()
        self.position = nd.array(position_mat)
        input_units, middle_units = position_mat.shape

        self.noise = gluon.nn.Dropout(dropout)

        self.weight_first = self.params.get('weight_first', shape=(input_units, middle_units), init=MyInit(relation=self.position))
        self.noise2 = gluon.nn.Dropout(dropout)
        # self.bias_first = self.params.get('bias_first', shape=(middle_units,))
        self.second_layer = gluon.nn.Dense(output_units, activation='relu')

    def forward(self, x):
        new_x = self.noise(x)
        self.singure = self.weight_first.data() * self.position
        self.first_layer = nd.Activation(nd.dot(new_x, self.singure), act_type='relu')
        self.output = self.second_layer(self.noise2(self.first_layer))
        return self.output
