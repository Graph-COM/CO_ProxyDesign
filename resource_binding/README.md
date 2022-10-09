# Here are the steps to train the model, test the model in the resource allocation in circuit design.

## Problem Settings
In this problem, we take the computation graph with hundereds of nodes, each nodes represents an operation in the circuit. Some of the operation has two choices of the elctronic devices: the DSP (digital signal processor) or the LUT (look-up table). We need to run very long time high level synthesis (HLS) to know the actual DSP/LUT usage. An important problem is to balance the DSP/LUT usage amount by assigning each operation with different electronic devices ($X$). 
Thus we need to first learn AFF/CON proxies to learn the actual DSP/LUT usage amount given a certain assignment of the circuit $f(X;C)$. Then we use the learnt proxies as loss fucntions to guide our algorithm $\mathcal{A}_{\theta}$ to learn to optimize and find out the best selection of the AC calculators that would result in the smallest error.

## Steps
- Step 1.1: The least environment requirement for imprecise functional unit assignment in approximation computing: 

```Python
conda create -n proxyco python=3.7.11
conda activate proxyco
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch-sparse==0.6.11 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111   # this may take a while...
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch_geometric==1.7.2
```

if you install the environment packages above but has the error as follows:
```
‘NoneType‘ object has no attribute ‘origin‘
```
you could add the cuda.so libs into the folder of these packages that we uploaded to the folder ./environment_utils/cuda_libs to solve the problem.

- Step 1.2: Install the packages that we would use:
```Python
pip install pyyaml
pip install tensorboardx
```

- Step 2.1: Generate the training, validation, testing dataset for proxy $f$, (this is also the training/validation dataset for the optimizer $\mathcal{A}_\theta$):

Go to ./generate_dataset/, we provide two versions of the high level synthesis circuit generator. Follow either of the instructions to generate the graph.

Then go to './build_dataset', fill in the path to the generated data in /build_dataset/configs/config.yaml. Run

```Python
python build_train.py
python build_val.py
```
- Step 2.3: Generate the testing dataset for the optimization algorithm $\mathcal{A}_\theta$):

This follows the same step as above. Here we provide some of the testing cases in /build_dataset/case folder as demo, the data-preprocess demo is also in this folder.

This process take too much time, because we need to find out the optimal solution for each instance, we provide our generated instances, you could directly skip this step
Go to ./build_dataset, run:
```Python
python build_inference.py
```

- Step 3.1: train the AFF/CON proxy:

Go to the train_dsp_con folder, if you would like to train the CON proxy, run:
```Python
python train_dsp_con.py
```
For the proxy of LUT, go to the train_lut_con folder, run:
```Python
python train_lut_con.py
```

- Step 3.2: train $\mathcal{A}_{\theta}$:

Go to the /train_atheta folder, if you would like to train $\mathcal{A}_{\theta}$ for the CON proxy, run
```Python
python train_a_con.py
```

- Step 4.1: Test with our learnt model to find the best DSP/LUT assignment!
Go to the /evaluation foler, run
```Python
sh test_con.sh
```
Before running the script, first set the save path in the code to save the assignment results.

- Step 4.2: Run high level synthesisi (HLS) to evaluate the actual DSP/LUT usage of each assignment.
Go back to generate_dataset, follow the instructions to evaluate the actual usage with HLS tools.



