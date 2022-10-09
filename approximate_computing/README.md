# approximate_computing
Here are the steps to train the model, test the model in the imprecise functional unit assignment in approximation computing.

## Problem Settings
In this problem, we take the computation graph with $15 = (8+4+2+1)$ nodes, each nodes represents an operation (either addition or multiplication). Each operation has two versions of calculators: the precise calculator or the approximate computing (AC) calculator. The former outputs accurate output, while the latter gives a $10\%$ relative error. To balance the computational complexity and efficiency, in real-world applications, there are always a threshold to set at least several ($3,5,8$ in our case) calculators to be AC (efficient but not accurate). The goal is to learn GNNs to assign the calculators whether to be precise or AC ($X$) such that the final relative error of the circuit could be as small as possible.

Thus we need to first learn AFF/CON proxies to learn the relative error given a certain assignment of the circuit $f(X;C)$. Then we use the learnt proxies as loss fucntions to guide our algorithm $\mathcal{A}_{\theta}$ to learn to optimize and find out the best selection of the AC calculators that would result in the smallest error.

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

Go to ./build_dataset, run
```Python
python build_data.py
```

- Step 2.2: Generate the testing dataset for the optimization algorithm $\mathcal{A}_\theta$):

This process take too much time, because we need to find out the optimal solution for each instance, we provide our generated instances, you could directly skip this step
Go to ./build_dataset, run:
```Python
python build_test.py
```

- Step 3.1: train the AFF/CON proxy:

Go to the train_proxy folder, if you would like to train the AFF proxy, run:
```Python
sh train_aff.sh
```
Or if you would like to train the CON proxy, run:
```Python
sh train_con.sh
```

- Step 3.2: train $\mathcal{A}_{\theta}$:

Go to the /train_atheta folder, if you would like to train $\mathcal{A}_{\theta}$ for the AFF proxy, run
```Python
sh train_a_aff.sh
```
Or if you would like to train the AFF proxy, run:
```Python
sh train_a_con.sh
```

- Step 4.1: Test with the learnt model to find the minimum weighted sum of edge covering!
Go to the /test foler, run
```Python
sh test_aff.sh
```
Or run
```Python
sh test_con.sh
```

You could find the tetsing results logs in the /test folder.



