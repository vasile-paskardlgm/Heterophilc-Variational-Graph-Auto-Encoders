import torch
import data_preprocessing as data_preprocess
import tools as tool
from Graph_Network_LGM import VGAEencoder

#torch.manual_seed(12346)
## Cuda selection
use_gpu = False

## Record of results
res_train = []
res_valid = []
res_test = []
loss_value = []


for _ in range(10):
    ## Use Cuda or not
    device = torch.device('cuda:0' if (use_gpu & torch.cuda.is_available()) else 'cpu')

    num_node,feat,posidx,negidx,adj,train_adj,train_posidx,train_negidx,test_posidx,test_negidx,valid_posidx,valid_negidx,train_nlap,label = data_preprocess.preprocess(dataset_name='Texas',\
        neg_rat=1,train_rat=0.85,test_rat=0.1)
    print("The edge rates of the dataset used now is: ")
    print(posidx.shape[1]/(num_node**2-num_node))

    model = VGAEencoder(feat.shape[1],128)       ##===================Baseline

    ## Move to gpu
    model = model.to(device)
    feat,posidx,negidx,adj,train_adj,train_posidx,train_negidx,test_posidx,test_negidx,valid_posidx,valid_negidx,train_nlap,label = feat.to(device),\
        posidx.to(device),negidx.to(device),adj.to(device),train_adj.to(device),train_posidx.to(device),train_negidx.to(device),test_posidx.to(device),\
            test_negidx.to(device),valid_posidx.to(device),valid_negidx.to(device),train_nlap.to(device),label.to(device)

    print('Now start training')
    
    ## Original Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    ## lr_Adjustment
    ## Learning_rate function, epoch:training epoch; [linear increase] + [exponential decrease]
    #lr_lambda = lambda epoch: (epoch+1)/50 if epoch<=49 else 2-(1+1e-5)**(epoch-50)

    ## Dynamic adjustment
    #schedular = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,last_epoch=-1,lr_lambda=lr_lambda,verbose=True)

## No Training process when range(0).
## In training processes,every epoch must neg_samp in order to learn more negative information.
    for epoch in range(100):
        optimizer.zero_grad()
        if epoch%1==0:
            with torch.no_grad():
                model.eval()
                _,_,z = model(feat,train_posidx)
                print("Train result: " , 100 * tool.test_inp(z=z,pos_edge_index=train_posidx)[0])
                print("Valid result: " , 100 * tool.test_inp(z=z,pos_edge_index=valid_posidx)[0])
                print("Test result: " , 100 * tool.test_inp(z=z,pos_edge_index=test_posidx)[0])
                print("Epoch: " , epoch)
        #if epoch%20==0:    
        #    stop = input("Stop?Pause buttom 1 to stop")
        #    if stop==str(1):
        #        break
        
        model.train()
        
        mu , logstd , z = model(feat,train_posidx)
        z = z.to(device)

        ## Recontruction loss
        recon_loss = tool.recon_inploss(z, pos_edge_index=train_posidx)
        ## KL-divergence loss
        kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        ## Propagation loss
        smoothness_loss = tool.propagation_loss(z=z, train_idx=train_posidx, train_adj=train_adj.float(), feat=feat, k=5)

        print("recon_loss,kl_loss,smoothness_loss:")
        print(recon_loss,kl_loss,smoothness_loss)

        ## Why (1/num_node): Avoiding Posteior-Collapse
        loss = recon_loss + (1/num_node) * kl_loss + 20 * smoothness_loss ##When Variational
        #loss = recon_loss       ##When NO Variational
        loss_value.append(loss)

        #if (epoch%1)==0:
        #    print("Now loss is: " , loss)
        #    print("Reconstruction loss: " , recon_loss)
        #    print("KL loss: " , kl_loss)
        #    stop = input("Stop?")
        #    if stop == str(1):
        #        break

        loss.backward()

        ##When NO lr adjustment
        optimizer.step()

        ##When lr adjustment
        #schedular.step()


## The model.eval() is necessary,because of the torch.nn.module
    print('Now start testing')
    
## Note that only in testing do we need to fixed the edges.
    with torch.no_grad():
        model.eval()
        _,_,z = model(feat,train_posidx)
        #z = model(feat,train_posidx)   ## Graphconv
        #z = model.encode(feat,train_posidx)     ## Models in Pyg
        result = tool.test_inp(z=z,pos_edge_index=train_posidx,neg_edge_index=train_negidx)
        res_train.append(round(result[0]*100,3))
        print("Train result_test: " , res_train[-1])
        if valid_posidx!=None:
            result = tool.test_inp(z=z,pos_edge_index=valid_posidx,neg_edge_index=valid_negidx)
            res_valid.append(round(result[0]*100,3))
            print("Valid result_test: " , res_valid[-1])
        result = tool.test_inp(z=z,pos_edge_index=test_posidx,neg_edge_index=test_negidx)
        res_test.append(round(result[0]*100,3))
        print("Test result_test: " , res_test[-1])

res_train = torch.tensor(res_train)
print("The Train ROC_AUC result: ", (res_train.mean() , res_train.var()))

if valid_posidx!=None:
    res_valid = torch.tensor(res_valid)
    print("The Valid ROC_AUC result: ", (res_valid.mean() , res_valid.var()))

res_test = torch.tensor(res_test)
print("The Test ROC_AUC result: ", (res_test.mean() , res_test.var()))