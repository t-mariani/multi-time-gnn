# multi-time-gnn

Personal project to implement GNN (Graph neural network) for multivariates time series 

First version is based on : 

```bibtex
@misc{wu2020connectingdotsmultivariatetime,
      title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks}, 
      author={Zonghan Wu and Shirui Pan and Guodong Long and Jing Jiang and Xiaojun Chang and Chengqi Zhang},
      year={2020},
      eprint={2005.11650},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2005.11650}, 
}
```


### Things that we need to understand
What is the channel in the convolution layer? Not clear at all for me.

How to deal with the dimensions:
- dimension of the time is changing
- dimension of the channels is changing

### TO DO:
- [x] How to deal with the dimensions: dimension of the channels is changing? Reduction ? Or it increases
-> set inside config 
- [x] implementation convolution skip connections
- [x] dropout after time module
- [x] regarder output module -> cf A.3 Experimental Setup
- [ ] Revoir les MLP layer -> doit etre selon les channels/features et non les capteurs
- [ ] Normalisation des dataset : scale par capteur ? 
- [ ] Layernorm : pourquoi autant de paramètres ? 
- [ ] Training : create real epoch -> Dataloader
- [ ] Avoir les mêmes métriques : horizon 3, 6, 12, 24 
- [ ] créer train, val, test set pour logger

Bonus:
- [ ] Faire le training avec une sous partie du graphe 
- [ ] Utiliser hydra 


### Constant names

N: number of sensors
A: adjency matrix (represents the graph)
C: embedding dimension