import numpy as np
#import scipy
import matplotlib.pyplot as plt
#import sklearn as skl
import pandas
import torch
import torch_geometric.utils as util
import torch_geometric.datasets as datasets
import homo_preprocessing as hoprepcs
import hete_preprocessing as heprepcs

feat_hom, posidx_hom, adj_hom = hoprepcs.preprocess()
feat_het, posidx_het, adj_het = heprepcs.preprocess()