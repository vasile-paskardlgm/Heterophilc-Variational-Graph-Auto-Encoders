import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import reset,glorot
from torch_geometric.nn import GCNConv,APPNP,ChebConv


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


class VGAEencoder(torch.nn.Module):
    #
    '''
    A single layer VGAE. Based on the paper '2016-Variational Graph Auto-Encoders'
    '''
    def __init__(self, in_channels, out_channels):
        super(VGAEencoder, self).__init__()
        self.convmu = torch.nn.Linear(in_channels, out_channels) 
        self.convlogstd = torch.nn.Linear(in_channels, out_channels) 
        self.propagate = APPNP(K=1, alpha=0)        ## APPNP returns: { (1-α)S·X+αX }, it is not the most general MPNN-GNN.
        self.dropout = torch.nn.Dropout(p=0.3)
        VGAEencoder.reset_param(self)


    def reset_param(self):
        reset(self.convmu)
        reset(self.convlogstd)
    
    def reparam(self,mu,logstd):

        ## Now we should get the Zeta:={μz + cor-eps(·)Σz},(·) means kronecker-product
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x, edge_index):
        #
        ''' Skeleton: (X,A) -> {GCN(μ,Σ)} -> (μ,Σ,Zeta)'''

        ## GCNμ
        mu = self.convmu(x)
        mu = self.dropout(mu)
        mu = self.propagate(mu, edge_index)

        ## GCNσ
        logstd = self.convlogstd(x)
        logstd = self.dropout(logstd)
        logstd = self.propagate(logstd, edge_index)

        zeta = self.reparam(mu,logstd)

        return mu , logstd , zeta

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Model: VGAE'

class Filter_encoder(torch.nn.Module):
    #
    '''
    A Filter-encoder
    '''
    def __init__(self, in_channels, hid_channels, out_channels):
        super(Filter_encoder, self).__init__()
        self.conv = ChebConv(in_channels, hid_channels, K=25) 
        self.batch = torch.nn.BatchNorm1d(hid_channels)
        self.convmu = torch.nn.Linear(hid_channels,out_channels)
        self.convlogstd = torch.nn.Linear(hid_channels,out_channels)
        self.lrelu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        Filter_encoder.reset_param(self)

    def reset_param(self):
        reset(self.convmu)
        reset(self.convlogstd)
        reset(self.conv)
        reset(self.batch)

    def reparam(self,mu,logstd):

        ## Now we should get the Zeta:={μz + cor-eps(·)Σz},(·) means kronecker-product
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    
    def forward(self,x,edge_index):
        x = self.conv(x,edge_index)
        x = self.dropout(x)
        #x = self.batch(x)
        x = self.lrelu(x)

        mu = self.convmu(x)

        logstd = self.convlogstd(x)
        logstd = F.normalize(logstd,p=2,dim=1) * 1.8

        zeta = self.reparam(mu,logstd)

        return mu , logstd , zeta

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 128, cached=True)
        self.conv_mu = GCNConv(128, out_channels, cached=True)
        self.conv_logstd = GCNConv(128, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)