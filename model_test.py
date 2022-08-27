import torch
import hete_preprocessing as heprepcs
import homo_preprocessing as hoprepcs
import tools as tool
from Graph_Network_LGM import VGNAEencoder,GAEencoder

#torch.manual_seed(12346)
res_train = []
res_valid = []
res_test = []


for _ in range(100):
    num_node,feat,posidx,negidx,adj,train_adj,train_posidx,train_negidx,test_posidx,test_negidx,valid_posidx,valid_negidx,train_nlap = heprepcs.preprocess(dataset_name='amherst41',neg_rat=1,train_rat=0.85,test_rat=0.1)
    print("The edge rates of the dataset used now is: ")
    print(posidx.shape[1]/(num_node**2-num_node))

    ## SC-filter bank preprocessing:
    #fil_bank, train_eigva, train_eigve = spcfil.adpt_filter_bank(nlap=train_nlap,s=5)
    #feat = spcfil.adpt_result_bank(node_fea=feat,filter_bank=fil_bank,eigve=train_eigve) ## Now we get a robust/collaborative features.

###  Note:The GAE performs well(90%+)and stably when neg_edges are NOT uesd in testing|training at same time, BEST in NONE-NEGATIVE
###  Note:The VGAE performs well(90%+)when neg_edges are NONE in testing|validating, and neg_edges in training makes model stable.
    ##Wisconsin , neg_rat=1
    #model = GAEencoder(feat.shape[1],16)
    model = VGNAEencoder(feat.shape[1],128)       ##===================Baseline

    print('Now start training')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

## No Training process when range(0).
    for epoch in range(20):
        optimizer.zero_grad()
        #if epoch%1==0:
        #    with torch.no_grad():
        #        model.eval()
        #        _,_,z = model(feat,train_posidx)
        #        print("Train result: " , 100 * tool.test(z=z,pos_edge_index=train_posidx,neg_edge_index=train_negidx)[0])
        #        print("Valid result: " , 100 * tool.test(z=z,pos_edge_index=valid_posidx,neg_edge_index=valid_negidx)[0])
        #        print("Epoch: " , epoch)
        #        stop = input("Stop?")
        #        if stop==str(1):
        #            break
        
        model.train()
        
        mu , logstd , z = model(feat,train_posidx)

        #z = model(feat,train_posidx)   ## Graphconv
        #mu , logstd = model.encoder(feat,train_posidx)   ## Models in Pyg
        #z = model.encode(feat,train_posidx)  ## Models in Pyg
        ## Recontruction loss
        recon_loss = tool.recon_loss(z, pos_edge_index=train_posidx,neg_edge_index=train_negidx)
        ## KL-divergence loss
        kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

        ## Why (1/num_node): Avoiding Posteior-Collapse
        loss = recon_loss + (1/num_node) * kl_loss   ##When Variational
        #loss = recon_loss       ##When NO Variational

#        if (epoch%20)==0:
#            print("Now loss is: " , loss)
#            print("Reconstruction loss: " , recon_loss)
#            print("KL loss: " , kl_loss)
#            stop = input("Stop?")
#            if stop == str(1):
#                break


        loss.backward()
        optimizer.step()


## The model.eval() is necessary,because of the torch.nn.module
    print('Now start testing')
    

    with torch.no_grad():
        model.eval()
        _,_,z = model(feat,train_posidx)
        #z = model(feat,train_posidx)   ## Graphconv
        #z = model.encode(feat,train_posidx)     ## Models in Pyg
        result = tool.test(z=z,pos_edge_index=train_posidx,neg_edge_index=train_negidx)
        res_train.append(round(result[0]*100,3))
        print("Train result_test: " , res_train[-1])
        if valid_posidx!=None:
            result = tool.test(z=z,pos_edge_index=valid_posidx,neg_edge_index=valid_negidx)
            res_valid.append(round(result[0]*100,3))
            print("Valid result_test: " , res_valid[-1])
        result = tool.test(z=z,pos_edge_index=test_posidx,neg_edge_index=test_negidx)
        res_test.append(round(result[0]*100,3))
        print("Test result_test: " , res_test[-1])

res_train = torch.tensor(res_train)
print("The Train ROC_AUC result: ", (res_train.mean() , res_train.var()))

if valid_posidx!=None:
    res_valid = torch.tensor(res_valid)
    print("The Valid ROC_AUC result: ", (res_valid.mean() , res_valid.var()))

res_test = torch.tensor(res_test)
print("The Test ROC_AUC result: ", (res_test.mean() , res_test.var()))