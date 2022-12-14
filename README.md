# Heterophilc-Variational-Graph-Auto-Encoders
***One of the sub-problems of my big idea: "Bridging the gap between Graph Signal Processing and Graph Learning"***  
## Logs  
*(22/08/23) A weird result: Training-free VGAE with only a uniform-initialization performs good (0.77+ in roc_auc_score) in all datasets, especially in two highly heterophilc datasets.*  
*(22/08/24) We found a revised VGAE could arive good performance in all dataset except for two heterophilc graph. We also found that there are many **inequal comparison, benchmark, baseline and others** in the papers about link-prediction(**Even in most papers about GNN!**)*  
*(22/08/29) Our targets include: i)**tackling the posterior-collapse while training**; ii)**thinking about why TRAING-FREE model with uniform-initialization would be preforms well the dataset[Texas,Wisconsin]**; iii)**searching for a more sharp and non-uniform-blur distribution**; iv)**finding out the correlation of latent variants z and disentangling them**.*  
*(22/09/15) It seems no strong relationship between smoothness of label and snoothness of feature, which means they are independent with the prosepective of edge-wise structure information. several questions occur: i)**Does label enhance the performance? Why?** ii)**If no, what is the relationship between the two definitions of smoothness?** iii)**If yes, how to involve label into the solution? Task or Non-Task?** iv)**Is higher-order smoothness(or others metrics) a possible solution?***  
## Problem Definition 
Supposed there is a **Heterophilc graph:G(V,E,X)**, which means the edge(i,j) exists when node i and j are different in their labels. Note that there is another similar but not same definition in **Heterophilc graph:G(V,E,X)**:edge(i,j) exists when node i and j are different in their features.  
  
We think about link-prediction in such a heterophilc graph. Specially, we treat it as a graph signal processing problem and solve it with a perspective of spectral graph network. We will also talk about modeling the correlated latent variants z which denotes a joint distribution in graph.  
## Proposed Method  
## Baseline Model  
## Dataset  
*All datasets used in our project comes from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html)*  
[**Cora,Citation,PubMed**]From the paper [2016-Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861)  
[**Photo**]From the paper [2018-Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868)  
[**chameleon,squirrel,crocodile**]From the paper [2019-Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)  
[**amherst41,johnshopkins55**]From the paper [2021-Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods](https://arxiv.org/abs/2110.14446)  
[**Texas,Wisconsin,Cornell**]From the paper [2020-Geom-GCN: Geometric Graph Convolutional Networks](https://arxiv.org/abs/2002.05287)  
## References  
[2016-Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)  
[2017-Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking](https://arxiv.org/abs/1707.03815)  
[2019-Isolating Sources of Disentanglement in VAEs](https://arxiv.org/abs/1802.04942)  
[2020-Graph Convolutional Gaussian Processes for Link Prediction](https://arxiv.org/abs/2002.04337)  
[2020-Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)  
[2021-Variational Graph Normalized Auto-Encoders](https://arxiv.org/abs/2108.08046)  
[2022-Restructuring Graph for Higher Homophily via Learnable Spectral Clustering](https://arxiv.org/abs/2206.02386)  
