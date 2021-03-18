#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2020/9/5 10:10
# @FileName: run_.py
# @Software: PyCharm

from DCNet.build import ObjectiveDCnet
from DCNet.models import DCNet,DCNetNoise


if __name__ == '__main__':
    epoch = 5
    output_dir = 'test_output'
    dd = 'train_test_data.pkl'
    rd = 'input_output_data.pkl'
    # DCNet
    odc = ObjectiveDCnet(net_model=DCNet, data_dir=dd, relation_dir=rd, epoch=epoch)
    odc.prepareData()
    odc.buildDCnet()
    errors = odc.train()
    # plot errors
    # odc.plotErrors(errors)
    odc.saveNet(output_dir, extra=f'DCNet')

    # DCNetNoise
    for dp in [1, 3, 5]:
        odc = ObjectiveDCnet(net_model=DCNetNoise, data_dir=dd, relation_dir=rd, epoch=epoch)
        odc.prepareData()
        odc.net = odc.net_model(odc.position, odc.output_dim, dropout=dp/10.)
        odc.net.initialize()# init=init.Constant(1)
        errors = odc.train()
        # plot errors
        # odc.plotErrors(errors)
        odc.saveNet(output_dir, extra=f'DCNetnoise-D{dp}')

