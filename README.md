# Heterophilc-Variational-Graph-Auto-Encoders
***One of the sub-problems of my big idea: "Bridging the gap between Graph Signal Processing and Graph Learning"***  
## Logs  
*(22/08/28) We could arive good performance. However, we also found that there are many **inequal comparison, benchmark, baseline and others** in the papers about link-prediction(**Even in most papers about GNN!**)*  
*(22/08/29) Our targets include: i)**tackling the posterior-collapse while training**; ii)**thinking about why TRAING-FREE model with uniform-initialization would be preforms well the dataset[Texas,Wisconsin]**; iii)**searching for a more sharp and non-uniform-blur distribution**; iv)**finding out the correlation of latent variants z and disentangling them**.*  
## Problem Definition 
Supposed there is a **Heterophilc graph:G(V,E,X)**, which means the edge(i,j) exists when node i and j are different in their labels. Note that there is another similar but not same definition in **Heterophilc graph:G(V,E,X)**:edge(i,j) exists when node i and j are different in their features.  
  
We think about link-prediction in such a heterophilc graph. Specially, we treat it as a graph signal processing problem and solve it with a perspective of spectral graph network.
## Proposed Method  
## Baseline Model  
## Dataset  
**All datasets used in our project comes from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html)**  
[**Cora,Citation,PubMed**]From the paper [2016-Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861)  
[**Photo**]From the paper [2018-Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868)  
[**chameleon,squirrel,crocodile**]From the paper [2019-Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)  
[**amherst41,johnshopkins55**]From the paper [2021-Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods](https://arxiv.org/abs/2110.14446)  
[**Texas,Wisconsin,Cornell**]From the paper [2020-Geom-GCN: Geometric Graph Convolutional Networks](https://arxiv.org/abs/2002.05287)  
## References  
[2017-Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking](https://arxiv.org/abs/1707.03815)  
[2021-Variational Graph Normalized Auto-Encoders](https://arxiv.org/abs/2108.08046)  
[2020-Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)  
[2022-Restructuring Graph for Higher Homophily via Learnable Spectral Clustering](https://arxiv.org/abs/2206.02386)  
