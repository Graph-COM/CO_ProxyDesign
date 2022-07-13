# Unsupervised-Learning-for-Combinatorial-Optimization-with-Principled-Proxy-Design
The official implementation of the paper 'Unsupervised Learning for Combinatorial Optimization with Principled Proxy Design'.

Haoyu Wang, Nan Wu, Hang Yang, Cong Hao, and Pan Li.

![image](https://github.com/peterwang66/Unsupervised-Learning-for-Combinatorial-Optimization-with-Principled-Proxy-Design/blob/main/img/architecture.jpg)

# Introduction of the Framework - a Simplified Tutorial
Our framework is a general unsupervised framework that could be used in the following problems:

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

For example, in our feature-based edge covering problem, we directly write out the constraint 

$$ \textbf{Edge Covering Constraint:} \quad g_r(\bar{X};C) = \sum_{v\in V} \prod_{e:v\in e}(1-\bar{X}\_e) $$

To mention that, in the Erdos probabilistic methods paper, the max-clique and weighted-cut could be classified into this case. Once their objectives and constraints could be written out by hand, they could definitely be written out in the entry-wise concave(affine) way. 

**case 2** $f$ or $g$ is unknown:

Use neural network as the proxy $h$ to learn them, the structure of the networks could be constructed as:

* To learn a discrete function $h:\{0,1\}^{|V|}\times \mathcal{C}\rightarrow \mathbb{R}$, we adopt a GNN as the relaxed proxy of $h$. We first define a latent graph representation in $\mathbb{R}^F$ whose entries are all entry-wise affine mappings of $X$.  

$$ \textbf{Latent representation:} \quad \phi(\bar{X};C) = W +\sum_{v\in V} U_{v} \bar{X}\_{v} + \sum_{v,u \in V, (v,u) \in E} Q_{v,u} \bar{X}\_{v} \bar{X}\_{u}$$

where $W$ is the graph representation, $U_{v}$'s are node representations and $Q_{v,u}$ are edge representations. These representations do not contain $X$ and are given by GNN encoding $C$. Here, we consider at most 2nd-order moments based on adjacent nodes as they can be easily implemented via current GNN platforms. Then, we use $\phi$ to generate entry-wise affine \& concave proxies as follows.

   $$ \textbf{Entry-wise Affine Proxy (AFF):}\quad h_r^{\text{a}}(\bar{X};C) = \langle w^a, \phi(\bar{X};C)\rangle. \quad\quad $$
   
   $$ \textbf{Entry-wise Concave Proxy (CON):}\quad h_r^{\text{c}}(\bar{X};C) = \langle w^c, -\text{Relu}(\phi(\bar{X};C))\rangle + b, $$ 
   
where $w^a,w^c\in\mathbb{R}^F, b\in\mathbb{R}$ are learnt parameters and $w^c\geq0$ guarantees entry-wise concavity.

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
Python 3.7.1
torch 1.9.0
torch-cluster           1.5.9
torch-geometric         1.7.2
torch-scatter           2.0.8
torch-sparse            0.6.11
torch-spline-conv       1.2.1
pandas                  1.3.0
numpy                   1.20.3
torchvision             0.10.0
tqdm                    4.62.2

Optional but recommend:
numba                   0.55.0 (accelerate the dataset generation)
matplotlib              3.5.1 (visualization)


```

# Application I: Feature Based Edge Covering and Node Matching in Graphs

# Application II: Resource Allocation in Circuit Design

# Application III: Imprecise Functional Unit Assignment in Approximate Computing
