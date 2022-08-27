from re import X
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import reset,glorot
from torch_geometric.nn import GCNConv,APPNP


class GAEencoder(torch.nn.Module):
    #
    '''
    A two layers GCN-encoder.
    '''
    def __init__(self, in_channels, out_channels):
        super(GAEencoder, self).__init__()
        self.batch1 = torch.nn.BatchNorm1d(2 * out_channels)
        self.batch2 = torch.nn.BatchNorm1d(out_channels)
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning
        GAEencoder.reset_param(self)

    def reset_param(self):
        reset(self.batch1)
        reset(self.batch2)
        reset(self.conv1)
        reset(self.conv2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batch1(x)
        x = x.relu()
        x = self.conv2(x,edge_index)
        x = self.batch2(x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Vaniila_GAE' + self.__module__


class VGNAEencoder(torch.nn.Module):
    #
    '''
    A two layers VGAE. With Norm trick. Based on the paper '2021-Variational Graph Normalized Auto-Encoders'
    '''
    def __init__(self, in_channels, out_channels):
        super(VGNAEencoder, self).__init__()
        self.conv1 = torch.nn.Linear(in_channels, 2 * out_channels) 
        self.batch = torch.nn.BatchNorm1d(2 * out_channels)
        self.convmu = torch.nn.Linear(2 * out_channels, out_channels) 
        self.convlogstd = torch.nn.Linear(2 * out_channels, out_channels) 
        self.propagate = APPNP(K=1, alpha=0)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.act = torch.nn.LeakyReLU()
        VGNAEencoder.reset_param(self)


    def reset_param(self):
        reset(self.conv1)
        reset(self.convmu)
        reset(self.convlogstd)
        reset(self.batch)
    
    def reparam(self,mu,logstd):

        ## Now we should get the Zeta:={μz + cor-eps(·)Σz},(·) means kronecker-product
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x, edge_index):
        #
        ''' Skeleton: (X,A) -> {GCN} -> H -> {GCN(μ,Σ)} -> (μ,Σ,Zeta)'''

        ## GCNx
        x = self.conv1(x)
        #x = self.dropout(x)
        x = self.propagate(x, edge_index)
        x = self.batch(x)
        x = self.act(x)

        ## GCNμ
        mu = self.convmu(x)
        mu = self.propagate(mu, edge_index)

        ## GCNσ-norm
        logstd = F.normalize(self.convmu(x),p=2,dim=1) * 1.8
        logstd = self.propagate(logstd, edge_index)

        zeta = self.reparam(mu,logstd)

        return mu , logstd , zeta

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Model: VGNAE'
