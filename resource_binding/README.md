# To implement our framework in resource binding

## Dataset

1. To use the HLS tools to generate the dataset, go to the './generate_dataset' folder and follow the readme file. We provide two versions of the HLS tools, do whichever but make sure to test with the same HLS version.

(Here we provide some examples as the testing data in the folder './build_dataset/case'. Running with these examples could help skip the first data generation step by HLS tools)

2. With the generated data, put it in the folder './case', and then go to the './build_dataset/case' folder, run the code:

```
python data_preprocess.py
```
to pre-process the generated data by HLS tools.

3. Go to './build_dataset' folder, build the dataset that could be sent into PyTorch geometric platform:
```
python build_train.py
python build_test.py
```
By this, the data is pre-processed and could be run on PyTorch geometric platform.

## Training 

Go to the following files to train the proxies or the $\mathcal{A}\_{\theta}$.

|                                     |      folder     |
|:-----------------------------------:|:---------------:|
| train the proxy without constraints |      train_dsp, train_lut      |
|         train the CON proxy         |   train_dsp_linear, train_lut_linear  |
| train $\mathcal{A}_{\theta}$ with GS-Tr         |     train_g     |
|           train $\mathcal{A}_{\theta}$ for CON           |  train_g_linear |


## Testing

1. We first need to do the relaxation-and-rounding procedure with our learnt proxies

go to the './evaluation' folder to do evaluation.

2. We need to get the real value for the assignments obtained in step 1.

To use HLS tools to get the actual LUT/DSP usage amount given an assignment, 

go to the './generate_dataset' folder. Run read_info and Get_perf file to evaluate with the real simulation.



