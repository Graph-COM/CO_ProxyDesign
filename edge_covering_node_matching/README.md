# edge_covering_node_matching
Here are the steps to train the model, test the model and visualize the model output in the edge covering problem.

## Dataset Information
The dataset utilizes the images in MNIST dataset.

## Problem Settings
In this problem, we take 16 MNIST images as the node cinfiguration of the graph $C$, 

For each two adjacent images (nodes), there is an edge between them. The edges are optimization variables $X$, we aim to decide which edges to choose such that all the chosen edges could not only cover all the nodes, but also make the cost function $f(X;C)$ as small as possible. Note that we carefully design the function $f(X;C)$, and the models **do not know the particular structure of the function $f$ in prior**, the models could only learn the mapping of function $f$ from the training set.

Thus we need to first learn AFF/CON proxies to learn the mapping of $f$. Then we use the learnt proxies as loss fucntions to guide our algorithm $\mathcal{A}_{\theta}$ to learn to optimize and find out the best selection of the edges.

## Steps
Before we start, we provide our pre-trained AFF proxy and its $\mathcal{A}_{\theta}$, after setting up the environments, you could directly play with them by jumping to step 4.1.

- Step 1.1: The least environment requirement for edge covering: 

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
pip instal  matplotlib
```

- Step 2.1: Pre-process the MNIST dataset:

Go to ./build_dataset, run
```Python
python load_mnist.py
```

- Step 2.2: Generate the training, validation, testing dataset for proxy $f$, (this is also the training/validation dataset for the optimizer $\mathcal{A}_\theta$):

Go to ./build_dataset/configs/config.yaml, open the file, fill in the path that you would like to put the training data after the target_path.

Then go to './build_dataset', run

```Python
python build_data.py
```

- Step 2.3: Generate the testing dataset for the optimization algorithm $\mathcal{A}_\theta$):

This process take too much time, because we need to find out the optimal solution for each instance, we provide our generated instances, you could directly skip this step
Go to ./build_dataset, run:
```Python
python generate_test.py
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

- Step 4.1: Test and paint with out learnt model to find the minimum weighted sum of edge covering!
Go to the /test foler, run
```Python
sh test_aff.sh
```
Or run
```Python
sh test_con.sh
```

You could find the tetsing results logs and the visualization of the edge covering in each tetsing instance in /test/paint folder.
To use our pre-trained model in  /test/pre_trained_model, to play with the testing set, set the path in test_aff.sh.



