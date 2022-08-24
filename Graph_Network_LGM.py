import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import reset
from torch_geometric.nn.conv import ChebConv,GCNConv

class SCmlp(torch.nn.Module):
    #
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super(SCmlp, self).__init__()
        self.mlp1 = torch.nn.Linear(in_channels, 2 * out_channels)
        self.mlp2 = torch.nn.Linear(2 * out_channels, out_channels)
        self.batch1 = torch.nn.BatchNorm1d(2 * out_channels)
        self.batch2 = torch.nn.BatchNorm1d(out_channels)
        self.resnet1 = Parameter(torch.FloatTensor(in_channels, 2 * out_channels))
        self.resnet2 = Parameter(torch.FloatTensor(2 * out_channels, out_channels))
        SCmlp.reset_param(self)
    
    def reset_param(self):
        reset(self.mlp1)
        reset(self.mlp2)
        reset(self.batch1)
        reset(self.batch2)
        reset(self.resnet1)
        reset(self.resnet2)

    def forward(self,x):
        fx = self.mlp1(x)
        x = torch.matmul(x,self.resnet1)
        x = self.batch1(x + fx)
        #x = self.batch1(fx)    ## No resnet
        x = x.relu()
        fx = self.mlp2(x)
        x = torch.matmul(x,self.resnet2)
        x = self.batch2(x + fx)
        #x = self.batch2(fx)    ## No resnet
        x = x.relu()


        return x

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Module(s):\n' + 'MLP: ' + str(self.mlp1) + ',' + str(self.mlp2) + '\n' \
                + 'Batch_norm:' + str(bool(self.batch1)) + '\n' + 'Resnet:' + str(bool(self.resnet1))

class Mlpcheb_GAE(torch.nn.Module):
    #
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super(Mlpcheb_GAE, self).__init__()
        self.filter = SCmlp(in_channels, 2 * out_channels)
        self.chebconv = ChebConv(2 * out_channels, out_channels , K=16)
        self.batch = torch.nn.BatchNorm1d(out_channels)
        self.resnet = Parameter(torch.FloatTensor(2 * out_channels, out_channels))
        Mlpcheb_GAE.reset_param(self)

    def reset_param(self):
        reset(self.chebconv)
        reset(self.batch)
        reset(self.resnet)

    def forward(self,x,edge_index):
        x = self.filter(x)
        fx = self.chebconv(x,edge_index)
        x = torch.matmul(x,self.resnet)
        x = self.batch(x + fx)
        x = x.relu()


        return x

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Filter_GAE' + self.__module__

class Chebs_GAE(torch.nn.Module):
    #
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super(Chebs_GAE, self).__init__()
        self.chebconv1 = ChebConv(in_channels, 4 * out_channels , K=16)
        self.chebconv2 = ChebConv(4 * out_channels, out_channels , K=16)
        self.batch1 = torch.nn.BatchNorm1d(4 * out_channels)
        self.batch2 = torch.nn.BatchNorm1d(out_channels)
        #self.resnet1 = Parameter(torch.FloatTensor(in_channels, 4 * out_channels))
        #self.resnet2 = Parameter(torch.FloatTensor(4 * out_channels, out_channels))
        Chebs_GAE.reset_param(self)

    def reset_param(self):
        reset(self.chebconv1)
        reset(self.chebconv2)
        reset(self.batch1)
        reset(self.batch2)
        #reset(self.resnet1)
        #reset(self.resnet2)

    def forward(self,x,edge_index):
        fx = self.chebconv1(x,edge_index)
        #x = torch.matmul(x , self.resnet1)
        x = self.batch1(fx)
        x = x.relu()
        fx = self.chebconv2(x,edge_index)
        #x = torch.matmul(x , self.resnet2)
        x = self.batch2(fx)
        x = x.relu()

        return x

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Chebs_GAE' + self.__module__

class Chebs_VGAE(torch.nn.Module):
    #
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super(Chebs_VGAE, self).__init__()
        self.chebconv = ChebConv(in_channels, 2 * out_channels , K=16)
        self.chebconvmu = ChebConv(2 * out_channels, out_channels , K=8)
        self.chebconvlogstd = ChebConv(2 * out_channels, out_channels , K=8)
        self.batch1 = torch.nn.BatchNorm1d(2 * out_channels)
        self.batch2 = torch.nn.BatchNorm1d(out_channels)
        self.batch3 = torch.nn.BatchNorm1d(out_channels)
        Chebs_VGAE.reset_param(self)

    def reset_param(self):
        reset(self.chebconv)
        reset(self.chebconvmu)
        reset(self.chebconvlogstd)
        reset(self.batch1)
        reset(self.batch2)
        reset(self.batch3)

    def forward(self,x,edge_index):
        x = self.chebconv(x,edge_index)
        x = self.batch1(x)
        x = x.relu()
        mu = self.chebconvmu(x,edge_index)
        logstd = self.chebconvlogstd(x,edge_index)
        mu = self.batch2(mu)
        logstd = self.batch3(logstd)

        return mu , logstd

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Chebs_VGAE' + self.__module__

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
    A two layers VGAE.
    '''
    def __init__(self, in_channels, out_channels):
        super(VGAEencoder, self).__init__()
        self.batch1 = torch.nn.BatchNorm1d(2 * out_channels)
        self.batch2 = torch.nn.BatchNorm1d(out_channels)
        self.batch3 = torch.nn.BatchNorm1d(out_channels)
        self.conv1 = GCNConv(in_channels, 2 * out_channels,) # cached only for transductive learning
        self.convmu = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning
        self.convlogstd = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning
    
    def reset_param(self):
        reset(self.conv1)
        reset(self.convmu)
        reset(self.convlogstd)
        reset(self.batch1)
        reset(self.batch2)
        reset(self.batch3)
    
    def reparam(self,mu,logstd):

        ## Now we should get the Zeta:={μz + cor-eps(·)Σz},(·) means kronecker-product
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batch1(x)
        x = x.relu()
        mu = self.convmu(x,edge_index)
        #mu = self.batch2(mu)
        logstd = self.convmu(x,edge_index)
        #logstd = self.batch3(logstd)
        zeta = self.reparam(mu,logstd)

        return mu , logstd , zeta

    def __repr__(self):
        return self.__class__.__name__ + '\n' \
            + 'Vaniila_VGAE' + self.__module__