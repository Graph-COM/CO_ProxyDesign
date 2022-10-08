# Unsupervised-Learning-for-Combinatorial-Optimization-with-Principled-Proxy-Design
The official implementation of the paper 'Unsupervised Learning for Combinatorial Optimization with Principled Proxy Design'.

Haoyu Wang, Nan Wu, Hang Yang, Cong Hao, and Pan Li.

![image](https://github.com/Graph-COM/CO_ProxyDesign/blob/main/img/architecture.jpg)

# Introduction of the Framework - a Simplified Tutorial
Our framework is a general unsupervised framework which could extend the probabilistic framework in **Erdos goes neural**[<sup>1</sup>](#refer-anchor-1) from CO problems to PCO problems, as shown in the following figure:

|                                                   | Our framework |
|:-------------------------------------------------:|:-------------:|
|    LCO: learning for combinatorial optimization   |    &#9745;    |
|  PCO: the objective/constraints require learning  |    &#9745;    |
|      Binary optimization variable: X = {0,1}      |    &#9745;    |
| Non-binary optimization variable: X = {0,1,...,n} |   &#9745; *   |

\* The non-binary optimization variables could be formulated into the binary forms, (e.g.
Choosing from $(1,2,3)$ could be represented as a binary vector $(X_1,X_2,X_3)$ with a constraint $X_1 + X_2 + X_3 \leq 1$), and then solved with our
framework. But note that adding too many constraints might make the training process more difficult. We are working on the future work to solve this problem.

Here we introduce  steps to solve your CO problem!

## Step 1: find the problem
List the consiguration $C$ and the optimization variable $X$ of the problem,

## Step 2: construct the objective fr, and the constraints gr in entry-wise concave form
We need to construct the $f$ or $g$ in entry-wise concave structure with respect to the optimization variables, such that a performance guarantee could be achieved when we obtain a low loss. There could be two cases:

**case 1** $f$ or $g$ could be written out by hand:

In this case, our relaxation-and-rounding procedure would reduce to the same framework as the probabilistic framework introduced in **Erdos goes neural**[<sup>1</sup>](#refer-anchor-1) such as the max-clique and the weighted-cut problem. Once the objectives could be written out by hand, they could be written out in the entry-wise concave (affine) way due to their discrete property (See our Theorem 2).

For example, in our feature-based edge covering problem, we directly write out the constraint 

$$ \textbf{Edge Covering Constraint:} \quad g_r(\bar{X};C) = \sum_{v\in V} \prod_{e:v\in e}(1-\bar{X}\_e) $$

**case 2** $f$ or $g$ is unknown:

We need to first use neural networks as the proxy $h$ to learn them, the structure of the networks could be constructed as:

* To learn a discrete function $h:\{0,1\}^{|V|}\times \mathcal{C}\rightarrow \mathbb{R}$, we adopt a GNN as the relaxed proxy of $h$. We first define a latent graph representation in $\mathbb{R}^F$ whose entries are all entry-wise affine mappings of $X$.  

$$ \textbf{Latent representation:} \quad \phi(\bar{X};C) = W +\sum_{v\in V} U_{v} \bar{X}\_{v} + \sum_{v,u \in V, (v,u) \in E} Q_{v,u} \bar{X}\_{v} \bar{X}\_{u}$$

where $W$ is the graph representation, $U_{v}$'s are node representations and $Q_{v,u}$ are edge representations. These representations do not contain $X$ and are given by GNN encoding $C$. Here, we consider at most 2nd-order moments based on adjacent nodes as they can be easily implemented via current GNN platforms. Then, we use $\phi$ to generate entry-wise affine \& concave proxies as follows.

   $$ \textbf{Entry-wise Affine Proxy (AFF):}\quad h_r^{\text{a}}(\bar{X};C) = \langle w^a, \phi(\bar{X};C)\rangle. \quad\quad $$
   
   $$ \textbf{Entry-wise Concave Proxy (CON):}\quad h_r^{\text{c}}(\bar{X};C) = \langle w^c, -\text{Relu}(\phi(\bar{X};C))\rangle + b, $$ 
   
where $w^a,w^c\in\mathbb{R}^F, b\in\mathbb{R}$ are learnt parameters and $w^c\geq0$ guarantees entry-wise concavity.

In implementation, we could formulate the AFF proxy as follows (not limited to):

1. use GNN to encode the configuration C
1. Divide the learnt encoding into two parts (coefficient and bias to multiply with X) to construct the latent representation $\phi(\bar{X};C) = W +\sum_{v\in V} U_{v} \bar{X}\_{v} + \sum_{v,u \in V, (v,u) \in E} Q_{v,u} \bar{X}\_{v} \bar{X}\_{u}$
1. For each node with its latent representation, calculate the logarithmic function of the representation, use message passing to add the adjacent log latent representation together, then do exponential function to get the AFF proxy.

In implementation, we could formulate the CON proxy as follows (not limited to):

1. we could use a constant to minus the AFF latent proxy, then go through the Relu() function, and finally we send the output into another fully connected layer. We constrain the weights of the last fully connected layer to be greater or equal to 0 (torch.clamp() function).

**Note:** We may sum the AFF proxy and the CON proxy for better performance.

## Step 3: formulate the problem
Form the problem into the following form:

$$ \min_{X \in \{0,1\}^n} f(X;C) \ \ \text{s.t. } g(X;C) < 1$$

An example of the normalization of the constraint to follow the above form is: $(g(\cdot;C) - g_{\min})/(g_{\min}^+ - g_{\min})$, where $g_{\min}^+ = \min_{X \in \{0,1\}^n \backslash \Omega} g(X;C)$ and $g_{\min} = \min_{X \in \{0,1\}^n} g(X;C)$. They could be easily eatimated in practice.

## Step 4: write out the relaxed training loss function

$$ \min_{\theta} l_r(\theta;C) \triangleq f_r(\bar{X};C) + \beta g_r(\bar{X};C), \text{where} \bar{X} = \mathcal{A}\_{\theta}(C) \in [0,1]^n, \beta > 0 $$

## Step 5: train the randomized algorithm with the loss function above
Train $\mathcal{A}\_{\theta}$ with the loss function above.

# Environment Requirements
The following packages are required to install to implement our code:
```shell
conda create -n proxyco python=3.7.11
conda activate proxyco
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch-sparse==0.6.11 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111   # this may take a while...
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch_geometric==1.7.2

Optional but recommend:
pip install matplotlib
pip install pyyaml
pip install tensorboardx
```

# Application I: Feature Based Edge Covering and Node Matching in Graphs

# Application II: Resource Allocation in Circuit Design

# Application III: Imprecise Functional Unit Assignment in Approximate Computing


<div id="refer-anchor-1"></div>
[1][Erdos goes neural: an Unsupervised Learning Framework for Combinatorial Optimization on Graphs. Neurips 2020.](https://proceedings.neurips.cc/paper/2020/hash/49f85a9ed090b20c8bed85a5923c669f-Abstract.html) 
