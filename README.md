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
- [x] Revoir les MLP layer -> doit etre selon les channels/features et non les capteurs
- [x] Normalisation des dataset : scale par capteur ? 
- [ ] Layernorm : pourquoi autant de paramètres ? 
- [x] Training : create real epoch -> Dataloader
- [ ] Avoir les mêmes métriques : horizon 3, 6, 12, 24 
- [x] créer train, val, test set pour logger

Bonus:
- [ ] Faire le training avec une sous partie du graphe 
- [ ] Utiliser hydra 


### Constant names

N: number of sensors
A: adjency matrix (represents the graph)
C: embedding dimension


### The normalisation strategy

We have a multivariate time series data. We want to normalise it.

We have three dataset: training, test, validation. 

We are going to normalise the training dataset for each dimension like that: $\hat{y}_{train, i} = \frac{y_{train, i} - \overline{y}_{train, i}}{\sigma_{train, i}}$

Where $\overline{y}_{train, i}$ is the mean of the training dataset in the i th dimension and $\sigma_{train, i}$ is its standard deviation 

Then, during inference we will do for all the dimension i:

$
\hat{y}_{test, i} = \frac{y_{test, i} - \overline{y}_{train, i}}{\sigma_{train, i}}
$