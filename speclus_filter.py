import numpy as np
import torch
import torch_geometric.utils as util
import torch_geometric.datasets as datasets
import homo_preprocessing as hoprepcs
import hete_preprocessing as heprepcs
import matplotlib.pyplot as plt

def filter_bank():
    #
    '''
    A function used to define spectral-clustering filter bank, which is the same as the paper 
        '2022-Restructuring Graph for Higher Homophily via Learnable Spectral Clustering'
    '''

    return 0

N,feat_het, posidx_het, adj_het,train_adj_het,train_posidx_het,test_posidx_het,_,train_nlap_het = heprepcs.preprocess(dataset_name='chameleon')
