# To implement our framework in resource binding

## Dataset

1. To use the HLS tools to generate the dataset, go to the './generate_dataset' folder and follow the readme file. We provide two versions of the HLS tools, do whichever but make sure to test with the same HLS version.

2. With the generated data, put it in the folder './case', and then go to the './build_dataset' folder, to pre-process the data and build the dataset that could be sent into PyTorch geometric platform.

Here we provide some examples as the testing data in the folder './example_data'

## Training 

Go to the following files to train the proxies or the $\mathcal{A}\_{\theta}$.

|                                     |      folder     |
|:-----------------------------------:|:---------------:|
| train the proxy without constraints |      train_dsp, train_lut      |
|         train the CON proxy         |   train_dsp_linear, train_lut_linear  |
| train $\mathcal{A}_{\theta}$ with GS-Tr         |     train_g     |
|           train $\mathcal{A}_{\theta}$ for CON           |  train_g_linear |


## Testing

To use HLS tools to get the actual LUT/DSP usage amount given an assignment, 

First, go to the './evaluation' folder to do evaluation.

Second, go to the './generate_dataset' folder. Run read_info and Get_perf file to evaluate.



