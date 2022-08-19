import torch
import torch_geometric.utils as util

def train_test_valid_set(num_nodes,edge_index,train_rat:float=0.8,test_rat:float=0.2):
    #
    '''
    This function is used to solve the the problem in torch_geometric. The problem is that the train_test_split function 
        could just concat the input edge_index and ignore the direct,which causes the output result has multiple edges in node-pair.
    ***Note that the function is just for undirected graph.
    1.Parameters:
    num_nodes:int,should be the number of Node in the graph
    edge_index:tensor,should be a (2,E) matrix,which indicates the positve edges;
    train_rat:float,ratio of training edges,default 0.8;
    test_rat:float,ratio of testing edges,default 0.2;
    2.Returns:
    train_posidx:tensor,training edges of undirected graph;
    test_posidx:tensor,tesing edges of undirected graph;
    valid_posidx:tensor,validating edges of undirected graph,it would be type(None) if train+test=1;
    '''
    assert 0<=(train_rat+test_rat)<=1, 'The total ratio must be smaller than 1!'
    assert (1>=train_rat>=0)&(1>=test_rat>=0), 'The ratio must be non-negative!'
    
    ## Trans it into undirected graph
    posidx = util.to_undirected(edge_index=edge_index)

    ## Get the edge index
    adj = torch.sparse_coo_tensor(indices=posidx,values=torch.ones_like(posidx[0,:]),size=(num_nodes,num_nodes))
    adj = adj.to_dense()
    adj =torch.triu(adj)
    posidx = util.dense_to_sparse(adj=adj)[0]  ## It means all the edges

    ## Sampling
    posidx = posidx[:,torch.randperm(posidx.shape[1])]
    if (1-train_rat-test_rat)==0:
        train_posidx = posidx[:,:int(train_rat*posidx.shape[1])]
        test_posidx = posidx[:,int(train_rat*posidx.shape[1]):]
        valid_posidx = None
        ## to_undirected
        train_posidx = util.to_undirected(train_posidx)
        test_posidx = util.to_undirected(test_posidx)
    else:
        train_posidx = posidx[:,:int(train_rat*posidx.shape[1])]
        valid_posidx = posidx[:,int(train_rat*posidx.shape[1]):int((1-test_rat)*posidx.shape[1])]
        test_posidx = posidx[:,int((1-test_rat)*posidx.shape[1]):]
        ## to undirected
        train_posidx = util.to_undirected(train_posidx)
        test_posidx = util.to_undirected(test_posidx)
        valid_posidx = util.to_undirected(valid_posidx)


    return train_posidx,test_posidx,valid_posidx