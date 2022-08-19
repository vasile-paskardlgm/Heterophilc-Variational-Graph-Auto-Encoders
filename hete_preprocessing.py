import torch
import torch_geometric.utils as util
import torch_geometric.datasets as datasets
import tools

def preprocess(dataset_name:str='crocodile',normalization='sym',url:str='.\\',rm_selfloop:bool=True,
                    train_rat:float=0.8,test_rat:float=0.2):
    #
    '''
    A function used to preprocess the **UNDIRECTED** heterophilic graph data. Return feat, posidx, adj
    dataset_name:str,'chameleon'|'crocodile'|'squirrel',all datasets come from torch.geometric_datasets, depend on the paper 
        '2020-Multi-scale Attributed Node Embedding';
    1.Parameters:
    normlization:str,'sym'|'rw'|type(None),choose a method to process the Graph Laplacian;
    url:str,default is '.\',head url of the datasets;
    rm_selfloop:bool,whether removes the self-loop or not,default True because of link-prediction;
    train_rat:float,ratio of training edges,default 0.8;
    test_rat:float,ratio of testing edges,default 0.2;
    2.Returns:
    N_dataset:int,number of nodes;
    feat_dataset:tensor,nodes'feature;
    posidx_dataset:tensor,edges of undirected original graph;
    adj_dataset:tensor,adjacency of undirected original graph;
    train_adj_dataset:tensor,adjacency of undirected train graph;
    train_posidx_dataset:tensor,training edges of undirected graph;
    test_posidx_dataset:tensor,tesing edges of undirected graph;
    valid_posidx_dataset:tensor,validating edges of undirected graph,it would be type(None) if train+test=1;
    train_nlap:tensor,normalized laplacian of training set; 
    '''
    ## Download the dataset into Class(Data) and extract the data.
    if dataset_name=='crocodile':
        dataset = datasets.WikipediaNetwork(root=url,name=dataset_name,geom_gcn_preprocess=False)
    else:
        dataset = datasets.WikipediaNetwork(root=url,name=dataset_name)
    
    ## Extract the data.
    feat_dataset = dataset.data['x']   ## Features and Nodes number.
    N_dataset = feat_dataset.shape[0]

    if dataset.data.is_undirected():
        ## When the graph is undirected:
        posidx_dataset = dataset.data['edge_index']
    else:
        ## When the graph is directed:
        posidx_dataset = dataset.data['edge_index']
        posidx_dataset = util.to_undirected(edge_index=posidx_dataset)

    if rm_selfloop:
        posidx_dataset = util.remove_self_loops(posidx_dataset)[0]  ## To remove the self-loop

    ## split it into train/test(or add valid) set.All of them are undirected.
    train_posidx_dataset,test_posidx_dataset,valid_posidx_dataset = tools.train_test_valid_set(num_nodes=N_dataset,edge_index=posidx_dataset,train_rat=train_rat,test_rat=test_rat)

    ## Adjacency. Training set/Original set/Testing set/Validation set
    adj_dataset = torch.sparse_coo_tensor(indices=posidx_dataset,values=torch.ones_like(posidx_dataset[0,:]),size=(N_dataset,N_dataset))
    adj_dataset = adj_dataset.to_dense()
    train_adj_dataset = torch.sparse_coo_tensor(indices=train_posidx_dataset,values=torch.ones_like(train_posidx_dataset[0,:]),size=(N_dataset,N_dataset))
    train_adj_dataset = train_adj_dataset.to_dense()

    ##Graph Laplacian
    train_nlap = util.get_laplacian(edge_index=train_posidx_dataset,normalization=normalization,num_nodes=N_dataset)  ##正则化拉普拉斯矩阵，特征值限定为[0,2]，得到lapMat的index与weight
    train_nlap = torch.sparse_coo_tensor(indices=train_nlap[0],values=train_nlap[1],size=(N_dataset,N_dataset))  ##以填充方式稀疏生成
    train_nlap = train_nlap.to_dense()  ##变成非稀疏的0填充矩阵

    return N_dataset,feat_dataset, posidx_dataset, adj_dataset,train_adj_dataset,train_posidx_dataset,test_posidx_dataset,valid_posidx_dataset,train_nlap
