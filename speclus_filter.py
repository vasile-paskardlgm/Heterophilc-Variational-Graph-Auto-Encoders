import numpy as np
import torch
import matplotlib.pyplot as plt


def diff_filter(eigva,a:float,eps:float=1e-4,s:int=40,m:int=4):
    #
    '''
    This function is used to get the filter as showed in the paper below;
    1.Parameters:
    eigva:tensor,should be a small-to-big eigenvalue-vector in usual,also could be a range from 0 to 2;
    a:float,the extremum point(median frequency) of the band-pass filter;
    eps:float,could be a hypermeter;
    s:int,the scale of the filter;
    m:int,the order of the filter;
    2.Returns:
    filter:tensor,every element would be a SC-graph filter. Note that:
        1)the filter is just a filter-vector,you must torch.diag_embed(filter-vector) so that you could get the filter-matrix;
        2)the sort,the filter-vector corresponds to eigenvalue-vector from smallest to biggest;
    '''
    filter = (((s*(eigva-a*torch.ones_like(eigva)))**(2*m))/((2+eps)**(2*m))+torch.ones_like(eigva))**(-1)
    
    return filter

def adpt_filter_bank(nlap,eps:float=1e-4,s:int=20,m:int=4):
    #
    '''
    A function used to define spectral-clustering filter bank, which is the same as the paper 
        '2022-Restructuring Graph for Higher Homophily via Learnable Spectral Clustering'
    1.Parameters:
    nlap:tensor,it must be normalized graph laplacian because eigva in range(0,2);
    eps:float,could be a hypermeter;
    s:int,the scale of the filter;
    m:int,the order of the filter;
    2.Returns:
    filter_bank:tuple(tensor),every element would be a SC-graph filter. Note that:
        1)the filter is just a filter-vector,you must torch.diag_embed(filter-vector) so that you could get the filter-matrix;
        2)the sort,the filter-vector corresponds to eigenvalue-vector from smallest to biggest;
    '''

    ## We first calculate the width between the filters and number of the filters
    rang = torch.arange(start=0,end=2,step=2e-4)
    zero = torch.zeros_like(rang)
    filter = diff_filter(rang,a=1)
    ## We set the width would be a range which makes a filter>0.5
    filter = torch.where(filter>0.5,filter,zero)
    width = (2e-4)*torch.abs(torch.nonzero(filter)[-1]-torch.nonzero(filter)[0])
    ## We could get the number of filters with range(0,2) and width
    filter_num = int(2/width)+1
    a = width*np.array(range(0,filter_num))
    ## So sorry about using eigen-decomposition, but the inv() in torch is fxxking disgust.
    eigva = torch.real(torch.linalg.eig(nlap)[0])
    eigve = torch.real(torch.linalg.eig(nlap)[1])
    eigva,sort = torch.sort(eigva)
    eigve = eigve[:,sort]
    filter_bank = [diff_filter(eigva=eigva,a=i,eps=eps,s=s,m=m) for i in a]

    return filter_bank, eigva, eigve

def dyna_draw(filter_bank,eigva):
    #
    '''
    This function is used to draw the filter bank you set in "adpt_filter_bank"
    1.Parameters:
    filter_bank:tuple(tensor),the filter bank you set and get from the function;
    nlap:tensor,must be normailized graph laplacian;
    2.Returns:
    None
    '''
    plt.title('Spectral Clustering Filter Bank')
    plt.xlabel('Frequency')
    plt.ylabel('Latitute')
    for filter in filter_bank:
        plt.scatter(eigva,filter,s=10)
        plt.pause(1.5)

def adpt_result_bank(node_fea,filter_bank,eigve):
    #
    '''
    This function is used to calculate the [filter:1,filter:2,...,filter:K](*)X -> [result:1,result:2,...,result:K]
    1.Parameters:
    filter_bank:tuple(tensor),the output of filter_bank generation function;
    eigve:tensor,eigenvalue and eigenvector;
    node_fea:tensor,nodes'features of the graph.
    2.Returns:
    X_bank:tensor,a bank of processed X,which can be seen as a collaborative classifier or a data-augmentation. 
        But, it is just a result of linear GNN(filter).
    '''
    result_bank = [torch.diag_embed(filter) for filter in filter_bank]
    result_bank = [torch.matmul(torch.matmul(eigve,filter),eigve.T) for filter in result_bank]
    result_bank = [torch.matmul(filter,node_fea) for filter in result_bank]

    ## Concate it,get a collaborative filter result:(N,FxK)
    result_bank = torch.concat(tensors=result_bank,dim=1)

    return result_bank
