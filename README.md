  # DCNet Package
  ### Using deep learning to unravel the cell profile from bulk expression data
  Deconvolution network (DCNet), an "explainable" artificial neural network model was created to infer cell content from bulk RNA-Seq samples. It embeds the relationship between 434 cells and 9078 marker genes in a neural network,and uses a hidden layer to characterize the cell type and content.
  # Publication
  Wang, X., Wang, H., Liu, D., Wang, N., He, D., Wu, Z., Zhu, X., Wen, X., Li, X., Li, J., & Wang, Z. (2022). Deep learning using bulk RNA-seq data expands cell landscape identification in tumor microenvironment. Oncoimmunology, 11(1), 2043662. https://doi.org/10.1080/2162402X.2022.2043662
  # Requirements and dependency
This package runs on Python 3.

DCNet depends on mxnet, pandas, numpy modules. You can install them by  `requirements`.

  # INSTALL
    git clone https://github.com/xindd/DCNet.git
    tar zxvf DCNet-0.0.1.tar.gz
    cd DCNet-0.0.1
    pip install -r requirements
    python setup.py install

   ## Dataset prepare
1.[DCNet params:](https://github.com/xindd/DCNet-Use/blob/main/net_M434_adam_T600_DCNetBnoise-D1.params) Contains DCNet parameters (weights, biases) saved based on `net.save_parameters(file_name)`.

To train the DCNet model, its parameters are set as adam optimizer, relu activation function, L2loss loss function, learning rate 1e-4, number of iterations 600, 256 samples per batch, cpu training respectively.

2.[Relation matrix:](https://github.com/xindd/DCNet-Use/blob/main/relation_matrix.pkl) A zero-one matrix containing the correspondence between genes and cells.
  
3.[Index](https://github.com/xindd/DCNet-Use/blob/main/index.pkl). Each neuron of DCNet corresponds to a specific gene. 
This file stores the set of genes corresponding to input and output neurons.

4.Input and output data matrix of gene expression profiles.
Each column is a sample, and each row should be a human gene symbol.
Please see [demo_data](https://github.com/xindd/DCNet-Use/blob/main/demo_data.pkl) as an example. A log2(x+1) transformation of the expression value is recommended.
   ## Running DCNet
   detail in [demo](https://github.com/xindd/DCNet-Use/blob/main/demo.ipynb)
   
    
