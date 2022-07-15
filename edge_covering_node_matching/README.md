# To implement our framework in edge covering and node matching

## Dataset
The dataset utilizes the images in MNIST dataset, the pre-processed form of the MNIST dataset is in the folder './mnist'.

- To generate the training dataset:

Fill in the path that you would like to put the training data in './build_dataset/configs/config'.

Enter './build_dataset', run

```Python
python build_data.py
```


- The test dataset is in the folder './create_testset/', the generating script is also included.

The testing script is also in this folder.

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

Fill in the path to the datasets and the models and train them.
## Testing

- Go to the create_test folder to do testing on the testset with the trained models.



