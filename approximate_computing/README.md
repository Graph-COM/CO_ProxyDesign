# To implement our framework in approximate computing

## Dataset
- To generate the training dataset:

Fill in the path that you would like to put the training data in './build_dataset/configs/config'.

Enter './build_dataset', run

```Python
python build_data.py
```


- The test dataset is in the folder './test/', the generating script is also included.

## Training

- To train the proxy $ f_r $ and the function $\mathcal{A}\_{\theta}$: 

|                                     |      folder     |
|:-----------------------------------:|:---------------:|
| train the proxy without constraints |      train      |
|         train the AFF proxy         |   train_linear  |
|         train the CON proxy         |  train_concave  |
| train $\mathcal{A}_{\theta}$ with GS-Tr         |     train_g     |
|           train $\mathcal{A}_{\theta}$ for AFF           |  train_g_linear |
|           train $\mathcal{A}_{\theta}$ for CON           | train_g_concave |


## Testing

- Go to the test folder to do testing on the testset with the trained models.
