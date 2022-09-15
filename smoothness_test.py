import torch
import data_preprocessing as data_preprocess
import torch.nn.functional as F
import torch_geometric.utils as util
import matplotlib.pyplot as plt

#
'''
This is used for testing the relationship between the smoothness of feature(GSP) and
    the smoothness of label(GL). We implement this experiment with the question: 
    " Whether both label and feature should be involved into heterophilc link-prediction? "
'''

## We choose eight dataset.
argset = ['Cora','CiteSeer','chameleon','squirrel','EN','DE','Texas','Wisconsin']
propogate = [0,1,2,3,4,5,10]     ## Define the number of propogation
rat = [0,0.05,0.1,0.15,0.2,0.5,0.7,0.9]

#pred_result = torch.tensor([93.21,96.37,94.27,93.53,86.83,88.03,90.13,73.12,81.31])
#no_train_result = torch.tensor([86.63,89.56,88.43,84.31,80.35,87.61,80.26,83.17,83.67])

## p-2 norm based distance metric:
def dist_smn(k,alpha):
    smoothness_f = []
    smoothness_l = []
    for dataset in argset:
        _,feat,idx,_,adj,_,_,_,_,_,_,_,_,label = data_preprocess.preprocess(dataset_name=dataset,normalization=None,neg_rat=1,train_rat=0.5,test_rat=0.5)
        label = F.one_hot(label)
        feat = feat.float()
        adj = adj.float()
        label = label.float()
        ## propagate
        for _ in range(k):
            feat = alpha * feat + torch.matmul(adj,feat)
            #label = alpha * label + torch.matmul(adj,label)
            feat = F.normalize(feat,2,1)
            #label = F.normalize(label,2,1)
        ## Calculate the smoothness
        lap = util.get_laplacian(edge_index=idx,normalization=None,num_nodes=feat.shape[0])
        lap = torch.sparse_coo_tensor(indices=lap[0],values=lap[1],size=(feat.shape[0],feat.shape[0]))  ##Generate a sparse matrix with index.
        lap = lap.to_dense()    ## Get laplacian
        smn_f = torch.matmul(torch.matmul(feat.T,lap),feat).trace()
        smn_f = smn_f/idx.shape[1]      ## Average distance
        smoothness_f.append(smn_f)
        smn_l = label[idx[0]] - label[idx[1]]
        smn_l = (smn_l * smn_l).sum()
        smn_l = smn_l/(idx.shape[1])
        smoothness_l.append(smn_l)

    smoothness_f = torch.tensor(smoothness_f)
    smoothness_l = torch.tensor(smoothness_l)

    return smoothness_f,smoothness_l


## Cosine-siml based metric:
def cos_smn(k,alpha):
    smoothness_f = []
    smoothness_l = []
    for dataset in argset:
        _,feat,idx,_,adj,_,_,_,_,_,_,_,_,label = data_preprocess.preprocess(dataset_name=dataset,normalization=None,neg_rat=1,train_rat=0.5,test_rat=0.5)
        label = F.one_hot(label)
        feat = feat.float()
        adj = adj.float()
        label = label.float()
        ## propagate
        for _ in range(k):
            feat = alpha * feat + torch.matmul(adj,feat)
            #label = alpha * label + torch.matmul(adj,label)
            feat = F.normalize(feat,2,1)
            #label = F.normalize(label,2,1)
        ## Calculate the smoothness
        smn_f = (F.normalize(feat[idx[0]],p=2,dim=1) * F.normalize(feat[idx[1]],p=2,dim=1)).sum()
        smn_f = smn_f/idx.shape[1]
        smoothness_f.append(smn_f)
        smn_l = (F.normalize(label[idx[0]],p=2,dim=1) * F.normalize(label[idx[1]],p=2,dim=1)).sum()
        smn_l = smn_l/idx.shape[1]
        smoothness_l.append(smn_l)

    smoothness_f = torch.tensor(smoothness_f)
    smoothness_l = torch.tensor(smoothness_l)

    return smoothness_f,smoothness_l


##self-defined edges rate:
def edge_rate_smn(k,rat):
    smoothness_f = []
    smoothness_l = []
    for dataset in argset:
        _,feat,_,_,_,adj,idx,_,_,_,_,_,_,label = data_preprocess.preprocess(dataset_name=dataset,normalization=None,neg_rat=1,train_rat=rat,test_rat=1-rat)
        label = F.one_hot(label)
        feat = feat.float()
        adj = adj.float()
        label = label.float()
        ## propagate
        for _ in range(k):
            feat = 0.6 * feat + torch.matmul(adj,feat)
            #label = 0.6 * label +  torch.matmul(adj,label)
            feat = F.normalize(feat,2,1)
            #label = F.normalize(label,2,1)
        ## Calculate the smoothness
        lap = util.get_laplacian(edge_index=idx,normalization=None,num_nodes=feat.shape[0])
        lap = torch.sparse_coo_tensor(indices=lap[0],values=lap[1],size=(feat.shape[0],feat.shape[0]))  ##Generate a sparse matrix with index.
        lap = lap.to_dense()    ## Get laplacian
        smn_f = torch.matmul(torch.matmul(feat.T,lap),feat).trace()
        smn_f = smn_f/idx.shape[1]      ## Average distance
        smoothness_f.append(smn_f)
        smn_l = label[idx[0]] - label[idx[1]]
        smn_l = (smn_l * smn_l).sum()
        smn_l = smn_l/(idx.shape[1])
        smoothness_l.append(smn_l)

    smoothness_f = torch.tensor(smoothness_f)
    smoothness_l = torch.tensor(smoothness_l)

    return smoothness_f,smoothness_l


#
'''
What happen to smoothness if we gradually remove some edges?
''' 
#for k in propogate:
#    #plt.figure(figsize=(18,12))
#    for rate in rat:
#        smoothness_f,smoothness_l = edge_rate_smn(k,round(1-rate,2))
#        plt.scatter(smoothness_f,smoothness_l,label=str(round(1-rate,2)))
#        print('Rate of edges: ' , 1-rate)
#        print('smoothness in feature: ' , smoothness_f)
#        print('smoothness in label: ' , smoothness_l)
#    plt.legend()
#    plt.xlabel('smoothness_f')
#    plt.ylabel('smoothness_l')
#    plt.title('Different rate of edges with the 2-norm smoothness,k: ' + str(k))
#    plt.show()


#
'''
What happen to smoothness if we randomly sample the same number of edges?
''' 
#for k in propogate:
    #plt.figure(figsize=(18,12))
#    print('The K is: ' + str(k))
#    for time in range(20):
#        smoothness_f,smoothness_l= edge_rate_smn(k,round(0.5,2))
#        plt.scatter(smoothness_f,smoothness_l,label='time: '+str(time+1))
#        print('Times: ' , time)
#        print('smoothness in feature: ' , smoothness_f)
#        print('smoothness in label: ' , smoothness_l)
#    plt.legend()
#    plt.xlabel('smoothness_f')
#    plt.ylabel('smoothness_l')
#    plt.title('0.5 rate edges with the 2-norm smoothness,k: ' + str(k))
#    plt.show()



#
'''
What relationship between smn-label and smn-feature?
'''
for k in propogate:
#    plt.figure(figsize=(18,12))
    for alpha in range(2):
        smoothness_f,smoothness_l = dist_smn(k,round(0.55+alpha*0.05,2))
        plt.scatter(smoothness_f,smoothness_l,label=str(round(0.55+alpha*0.05,2)) + ':1')
        print('K: ' + str(k) + ' alpha: ' + str(round(0.55+alpha*0.05,2)))
        print('smoothness in feature: ' , smoothness_f)
        print('smoothness in label: ' , smoothness_l)
    plt.legend()
    plt.xlabel('smoothness_f')
    plt.ylabel('smoothness_l')
    plt.title('dist_product with propagation k: ' + str(k))
    plt.show()

#for k in propogate:
#    plt.figure(figsize=(18,12))
#    for alpha in range(2):
#        smoothness_f,smoothness_l = cos_smn(k,round(0.55+alpha*0.05,2))
#        plt.scatter(smoothness_f,smoothness_l,label=str(round(0.55+alpha*0.05,2)) + ':1')
#        print('K: ' + str(k) + ' alpha: ' + str(round(0.55+alpha*0.05,2)))
#        print('smoothness in feature: ' , smoothness_f)
#        print('smoothness in label: ' , smoothness_l)
#    plt.legend()
#    plt.xlabel('smoothness_f')
#    plt.ylabel('smoothness_l')
#    plt.title('cos_product with propagation k: ' + str(k))
#    plt.show()