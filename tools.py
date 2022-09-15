import torch
import torch_geometric.utils as util
import torch.nn.functional as F
from torch.linalg import det
from sklearn.metrics import average_precision_score, roc_auc_score

###########################################################
'''
This function is used to replace the torch.linalg.inv(),when we want to run GNN in cuda.
Because the cuda does not support torch.linalg.inv()
''' 
def cof1_cuda(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))
 
def alcof_cuda(M,index):
    return pow(-1,index[0]+index[1])*cof1_cuda(M,index)
 
def adj_cuda(M):
    result = torch.zeros((M.shape[0],M.shape[1]))
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof_cuda(M,[i,j])
    return result
 
def invmat_cuda(M):
    return 1.0/det(M)*adj_cuda(M)
###########################################################


def train_test_valid_set(num_nodes,edge_index,train_rat:float=0.85,test_rat:float=0.1,neg_rat:int=2):
    #
    '''
    This function is used to solve the the problem in torch_geometric. The problem is that the train_test_split function 
        could just concat the input edge_index and ignore the direct,which causes the output result has multiple edges in node-pair.
    ***Note that the function is just for undirected graph.
    1.Parameters:
    num_nodes:int,should be the number of Node in the graph
    edge_index:tensor,should be a (2,E) matrix,which indicates the positve edges;
    train_rat:float,ratio of training edges,default 0.85;
    test_rat:float,ratio of testing edges,default 0.1;
    neg_rat:int,ratio of (negative sampling : positive edges),default 2;
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

    ## Positive Sampling:
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

    ## Negative sampling:  you must mul the num with 2, otherwise, it will be neg:pos=0.5 
    negidx = util.negative_sampling(edge_index=posidx,num_nodes=num_nodes,num_neg_samples=2*neg_rat*(posidx.shape[1]),force_undirected=True)

    ## Get the all edges:
    adj = torch.sparse_coo_tensor(indices=negidx,values=torch.ones_like(negidx[0,:]),size=(num_nodes,num_nodes))
    adj = adj.to_dense()
    adj =torch.triu(adj)
    negidx = util.dense_to_sparse(adj=adj)[0]  ## It means all the edges

    ## Sampling:
    negidx = negidx[:,torch.randperm(negidx.shape[1])]
    if (1-train_rat-test_rat)==0:
        train_negidx = negidx[:,:int(train_rat*negidx.shape[1])]
        test_negidx = negidx[:,int(train_rat*negidx.shape[1]):]
        valid_negidx = None
        ## to_undirected
        train_negidx = util.to_undirected(train_negidx)
        test_negidx = util.to_undirected(test_negidx)
    else:
        train_negidx = negidx[:,:int(train_rat*negidx.shape[1])]
        valid_negidx = negidx[:,int(train_rat*negidx.shape[1]):int((1-test_rat)*negidx.shape[1])]
        test_negidx = negidx[:,int((1-test_rat)*negidx.shape[1]):]
        ## to undirected
        train_negidx = util.to_undirected(train_negidx)
        test_negidx = util.to_undirected(test_negidx)
        valid_negidx = util.to_undirected(valid_negidx)



    return train_posidx,train_negidx,test_posidx,test_negidx,valid_posidx,valid_negidx,negidx

def test_inp(z, pos_edge_index, neg_edge_index=None,neg_rat:int=1):
    """Given latent variables :obj:`z`, positive edges
    :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
    computes area under the ROC curve (AUC) and average precision (AP)
    scores.

    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to evaluate
            against.
        neg_edge_index (LongTensor): The negative edges to evaluate
            against.
        neg_rat (int): Rate of neg_edges:pos_edges
    """
    
    if neg_edge_index is None:
        ## you must mul the num with 2, otherwise, it will be neg:pos=0.5 
        neg_edge_index = util.negative_sampling(edge_index=pos_edge_index, num_nodes=z.size(0),
            num_neg_samples=2*neg_rat*pos_edge_index.shape[1],force_undirected=True)

    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    #neg_y = neg_y[:pos_y.size(0)]      ## To keep N_neg_edges == N_pos_edges
    
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))
    neg_pred = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
    #neg_pred = neg_pred[torch.randperm(neg_pred.size(0))]      ## To keep N_neg_edges == N_pos_edges
    #neg_pred = neg_pred[:pos_pred.size(0)]

    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred) 

def test_norm(z, pos_edge_index, neg_edge_index=None,neg_rat:int=1,normalization=True):
    """Given latent variables :obj:`z`, positive edges
    :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
    computes area under the ROC curve (AUC) and average precision (AP)
    scores.

    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to evaluate
            against.
        neg_edge_index (LongTensor): The negative edges to evaluate
            against.
        neg_rat (int): Rate of neg_edges:pos_edges
    """
    if normalization:
        z = F.normalize(z)

    if neg_edge_index is None:
        ## you must mul the num with 2, otherwise, it will be neg:pos=0.5 
        neg_edge_index = util.negative_sampling(edge_index=pos_edge_index, num_nodes=z.size(0),
            num_neg_samples=2*neg_rat*pos_edge_index.shape[1],force_undirected=True)

    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = z[pos_edge_index[0]] - z[pos_edge_index[1]]
    pos_pred = 2 * torch.sigmoid(-1 * (pos_pred * pos_pred).sum(dim=1))
    neg_pred = z[neg_edge_index[0]] - z[neg_edge_index[1]]
    neg_pred = 2 * torch.sigmoid(-1 * (neg_pred * neg_pred).sum(dim=1))

    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)

def recon_inploss(z, pos_edge_index, neg_edge_index=None, neg_training=True, neg_rat:int=1):
    """Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.

    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to train against.
        neg_edge_index (LongTensor, optional): The negative edges to train
            against. If not given, uses negative sampling to calculate
            negative edges.
        neg_training (bool): Whether uses negative_edges in training or not.
        neg_rat (int): Rate of neg_edges:pos_edges
    """
    eps = 1e-15

    pos_loss = -torch.log(
        torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)) + eps).mean()

    if neg_training:
        if neg_edge_index is None:
            ## you must mul the num with 2, otherwise, it will be neg:pos=0.5 
            neg_edge_index = util.negative_sampling(edge_index=pos_edge_index, num_nodes=z.size(0),
                num_neg_samples=2*neg_rat*pos_edge_index.shape[1],force_undirected=True)
        neg_loss = -torch.log(1 -torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + eps).mean()
    else:
        neg_loss = 0

    return pos_loss + neg_loss

def recon_normloss(z, pos_edge_index, neg_edge_index=None, neg_training=True, p=2, neg_rat:int=1,normalization=True):
    """Given latent variables :obj:`z`, computes the p-norm loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.

    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to train against.
        neg_edge_index (LongTensor, optional): The negative edges to train
            against. If not given, uses negative sampling to calculate
            negative edges.
        neg_training (bool): Whether uses negative_edges in training or not.
        p (float): the order of p-norm
        neg_rat (int): Rate of neg_edges:pos_edges
    """
    eps = 1e-15

    if normalization:
        z = F.normalize(z)

    pos_pred = z[pos_edge_index[0]] - z[pos_edge_index[1]]
    pos_pred = 2 * torch.sigmoid(-1 * (pos_pred * pos_pred).sum(dim=1))

    pos_loss = -torch.log(pos_pred + eps).mean()

    if neg_training:
        if neg_edge_index is None:
            ## you must mul the num with 2, otherwise, it will be neg:pos=0.5 
            neg_edge_index = util.negative_sampling(edge_index=pos_edge_index, num_nodes=z.size(0),
                num_neg_samples=2*neg_rat*pos_edge_index.shape[1],force_undirected=True)
        neg_pred = z[neg_edge_index[0]] - z[neg_edge_index[1]]
        neg_pred = 2 * torch.sigmoid(-1 * (neg_pred * neg_pred).sum(dim=1))
 
        neg_loss = -torch.log(neg_pred + eps).mean()
    else:
        neg_loss = 0

    return pos_loss + neg_loss

def propagation_loss(z,train_idx,feat,k:int=5,train_adj=None):
    #
    '''
    The propagation loss, which calculates the smoothness of feature
    1.Parameters:
    z:Tensor,nodes' embedding;
    train_idx:LongTensor,the index tensor of positive training edges;
    feat:Tensor,the nodes'feature;
    k:int,the highest order of propagation in adjacency matrix, default=5
    train_adj:Tensor,the adjacency matrix of positive training edges, default=None
    2.Return:
    smoothness:Tensor,smoothness loss.
    '''
    ## Set the self-loop factor
    alpha = 0.6

    ## Calculate the adjacency of positive training edges(if not give)
    if train_adj == None:
        train_adj = torch.sparse_coo_tensor(indices=train_idx,values=torch.ones_like(train_idx[0,:]),size=(feat.shape[0],feat.shape[0]))
        train_adj = train_adj.to_dense()
        train_adj = train_adj.float()

    ## Calculate the recon_adj
    recon_adj = torch.matmul(z,z.T).sigmoid().float()

    ## Calculate te recon_idx
    recon_idx = recon_adj.to_sparse()

    ## Get laplacian
    lap_recon = util.get_laplacian(edge_index=recon_idx.indices(),edge_weight=recon_idx.values(),normalization=None,num_nodes=feat.shape[0])
    lap_recon = torch.sparse_coo_tensor(indices=lap_recon[0],values=lap_recon[1],size=(feat.shape[0],feat.shape[0]))  ##Generate a sparse matrix with index.
    lap_recon = lap_recon.to_dense().float()

    lap_train = util.get_laplacian(edge_index=train_idx,normalization=None,num_nodes=feat.shape[0])
    lap_train = torch.sparse_coo_tensor(indices=lap_train[0],values=lap_train[1],size=(feat.shape[0],feat.shape[0]))  ##Generate a sparse matrix with index.
    lap_train = lap_train.to_dense().float()

    smoothness = 0

    for time in range(3,k+1):
        ## propagate
        x = feat.clone()
        y = feat.clone()
        for _ in range(time):
            x = alpha * x + torch.matmul(recon_adj,x)
            x = F.normalize(x,2,1)
            y = alpha * y + torch.matmul(train_adj,y)
            y = F.normalize(y,2,1)

        ## Calculate the smoothness
        smn_recon = torch.matmul(torch.matmul(x.T,lap_recon),x).trace()
        smn_recon = smn_recon/(recon_idx.values().sum())
        smn_train = torch.matmul(torch.matmul(y.T,lap_train),y).trace()
        smn_train = smn_train/(train_idx.shape[1])

        smoothness = smoothness + (smn_train - smn_recon).abs()

    return smoothness