  # DCNet Package
  ### Using deep learning to unravel the cell profile from bulk expression data
  Deconvolution network (DCNet), an "explainable" artificial neural network model wascreated to infer cell content from bulk RNA-Seq samples. It embeds the relationship between 434 cells and 9078 marker genes in a neural network,and uses a hidden layer to characterize the cell type and content.
    
  # INSTALL
    tar zxvf DCNet-0.0.1.tar.gz
    cd DCNet-0.0.1
    python setup.py install
    
   # Use DCNet
  ```python
    from DCNet.build import ObjectiveDCnet
    from DCNet.models import DCNet,DCNetNoise
    import os
    
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
  ```
  # Fine train DCNet
  ```python
    import pandas as pd
    import pickle
    from mxnet import nd, gluon
    from DCNet.dcutils import loadPosition, fineTrain, splitProfile, abline
    from DCNet.models import *
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    if __name__ == '__main__':
        print('load net...')
        net_name =  f'net_M434_adam_T600_DCNetnoise-D1.pkl'
        net_dir = net_name
        rd = 'input_output_data.pkl'
    
        with open(net_dir, 'rb') as f:
            save_data = pickle.load(f)
    
        net = save_data['net']
        ix = save_data['input_ix']
        ox = save_data['output_ix']
        _, _, position = loadPosition(rd)
    
        # load data from CRC
        crc_bulk_dir = 'CRC_bulk.csv'
        crc_cell_dir = 'CRC_cells.csv'
        crc_fratcion = 'CRC_fraction.csv'
        CRC_bulk = pd.read_table(crc_bulk_dir, sep=',', index_col=0)
        CRC_cells = pd.read_table(crc_cell_dir, sep=',', index_col=0)
        CRC_fraction = pd.read_table(crc_fratcion, sep=',', index_col=0)
        CRC_bulk = CRC_bulk.T
    
        x, x_na = splitProfile(CRC_bulk, ix)
        y, y_na = splitProfile(CRC_bulk, ox)
        x = nd.array(np.log2(x + 1).T)
        y = nd.array(np.log2(y + 1).T)
    
        trainer_1= gluon.Trainer(net.collect_params('.*weight_first'), 'adam', {'learning_rate': 1e-3, 'wd': 0.01})
        trainer_2 = gluon.Trainer(net.collect_params('dense.*'), 'adam', {'learning_rate': 1e-3, 'wd': 0.01})
        loss_fun = gluon.loss.L2Loss()
        fineTrain(net, (x, y, y_na), trainer_1, trainer_2, loss_fun, epochs=5)
    
        loss = net(x)
        output_y = net.output
        cell_purity = net.first_layer
        cp = cell_purity.asnumpy()
        # cell fraction
        cp = pd.DataFrame(cp, columns=position.columns, index=CRC_bulk.columns)
        ##
        plt.scatter(y.asnumpy().flatten(), output_y.asnumpy().flatten())
        abline(1,0)
        plt.show()
```
   
# More code
   <https://github.com/xindd/DCNet-Use>
    
