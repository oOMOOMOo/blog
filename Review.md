# Machine Learning
## Fundamental

  ### Probability Theory (lecture 1)

  - Basic
    - Sum rule
      - $p(X=x_i)=\frac{c_i}{N}=\frac{1}{N}\sum_{j=1}^Ln_{ij}$

      概率和=所有概率相加

    -  Product rule
      -  $p(X=x_i,Y=y_i)=\frac{n_{ij}}{n}=\frac{n_{ij}}{c_i}\frac{c_i}{N}=p(Y=y_i|X=x_i)p(X=x_i)$

      概率乘积= A发生的概率 * 在A发生概率下B发生的概率

    - Joint probability
      - $p(X=x_i,Y=y_i)=\frac{n_{ij}}{N}$

    - Marginal probability
      - $p(X=x_i)=\frac{c_i}{N}$

    - Conditional probability
      - $p(Y=y_i|X=x_i)=\frac{n_{ij}}{c_i}$

  - Probability Densities
    - $p(x\in (a,b))= \int_a^b p(x)dx$   
    
      Probabilities over `continuous variables` are defined over their `probability density function (pdf)$p(x)$`

  - Expectations
    - discretes
      - $Formular $
      - 
    - continuous
      - $Fomular $
    - conditional expectation
    
  ![expectations](assert/ML_1_Expectations.png)

      The average value of some function $f(x)$ under a probability distribution $p(x)$ is called its `expectation`

  - Variances

  ![variances](assert/ML_1_Variances_and_Covariances.png)

  The variance provides a measure how much variability there is in $f(x)$ around its mean value $\mathbb{E}   f(x)$ .
  - Covariances


  ### Bayes Decision Theroy (Lecture 2)
    
    - Basic concepts
      - Priors
        - $\sum_kp(C_k)=1$
      - Conditional probabilities
        - $p(x|C_k)$
      - Posterior probabilites
        - $p(C_k|x)$
    - Minimizing the misclassification rate
    - Minimizing the expected loss
    - Discriminant functions
  ### Probability Density Estimation (Lecture 3)
  - General concepts
  - Gaussian distribution
    #### Parametric Methods
    - Maximum Likelihood approach
    - Bayesian vs. Frequentist views on probability
    - Bayesian Learning
    #### Non-Parametric Methods (Lecture4)
    - Histograms
    - Kernel density estimation
    - K-Nearest Neighbors
    - k-NN for Classification
    - Bias-Variance tradeoff

    #### Mixture distributions (Lecture 5)
    - Mixture of Gaussians (MoG)
    - Maximum Likelihood estimation attempt

    #### `K-Means Clustering`
    - Algorithm
    - Applications
    #### `EM Algorithm`
    - Credit assignment problem
    - MoG estimation
    - EM Algorithm
    - Interpretation of K-Means
    - Technical advice

## Classification Approaches

### Linear Discriminants (Lecture 6)

#### Linear discriminant funcions
- Definition
  - Bayesian Decision Theory $p(C_k|x)=\frac{p(x|C_k)p(C_k)}{p(x)}$
    - $p(x|C_k)$
    - $p(C_k)$: priors
    - $p(C_k|x)$: posteriors (need compute using Bayes'rule)
    - Minimize probability of misclassification by **maximizing** $p(C|x)$
  - New approach
    - Directly encode decision boundary
    - Without explicit modeling of probability densitites
    - Minimize misclassification probability directly

---
  - Goal: take a new input $x$ and assign it to one of $k$ classes $C_k$通过一个方程来判断应该归为哪类
  - Given:
    - training set
    - target set
    - General classification problem
    - 2-class problem
      - $y(x)=w^Tx+\omega_0$
        - $w$: weight vector
        - $\omega_0$: "bias" (=threshold)
      - If a data set can be **perfectly** classified by a linear discriminant, then we call it **linearly separable**
      
      $y(x)=0$ defines a hyperplane
      - Normal vector: $w$
      - Offset: $\frac{-\omega_0}{||w||}$
    
    - K-class problem
  - 
- Extension to multiple classes
  - Starategy 
    - One-vs-all
    - One-vs-one
      ![multipul_class](assert/ML_6_Multiple_Classe.png)
  - Problem
    - ambiguous: can't define all region
  - Solution
    - Taking K Linear function of the form $y_k(x)=W_k^Tx+\omega_{k_{0}}$


#### Least-squares classification
- Derivation
- Shortcomings
  - very sensitive to outliers
  - Reason: Least-squares corresponds to **Maximum Likelihood** under the assumption of a **Gaussian** conditional distribution. but some binary target vectors can have non-Gaussian distribution.

#### Generalized linear models (Lecture 7.1)
- Connection to neural networks
- Generalized linear discriminants & gradient descent

#### ***`Logistic Regression`*** (Lecture 7.2)
- Probabilistic discriminative models
- Logistic sigmoid (logit function)
- Cross-entropy error
- Iteratively Reweighted Least Squares

#### ***Softmax Regression***
- Multi-class generalization
- Gradient descent solution

#### Note on Error Funcions (not relevant?)

### Support Vector Machines (Lecture 8)
#### Motivation
#### Lagrangian (primal) formulation
#### Dual formulation

#### ***Soft-margin classification***

#### Nonlinear Support Vectore Machines (Lecture 9)
##### Nonlinear basis functions
##### ***The Kernel trick***
##### Mercer's condition
##### Popular kernels
#### Analysis
- Error function

#### Applications

### Ensemble Methods & Boosting (Lecture 10/11)
#### Analyzing Error Functions
#### Ensembles of classifiers
- Bagging
- Bayesian Model Averaging
#### `AdaBoost`
- Intuition
- Algorithm
- Analysis
- Extensions

### Randomized Trees, Forests & Ferns

## Deep Learning
### Foundations (Lecture 12)
#### A Brief History of Neural Networks
#### Perceptrons
- Definition
- Loss functions
- Regularization
- Limits
#### Multi-Layer Perceptrons (Lecture 13)
- Definition
- Learning with hidden units
#### Learning Multi-layer Networks
- Naive analytical differentiation
- Numerical differentiation
- Backpropagation
- Computational graphs & Automatic differentiation
- Practical issues

#### Gradient Descent (Lecture 14)
- Stochastic Gradient Descent & Minibatches
- Choosing Learning Rates
- Momentum
- RMS Prop
- Other Optimizer/Effect of optimizer

#### Tricks of the Trade (Lecture 15)
- Shuffling
- Data Augmentation
- Normalization
- Nonlinerities
- Initialization
- Dropout
- Batch Normalization
#### Advanced techniques

### `Convolutional Neural Networks`
#### CNN
- Neural Networks for Computer Vision
- Convolutional Layers
- Pooling Layers

#### CNN Architectures (Lecture 16)
- LeNet
- AlexNet
- VGGNet
- GooLeNet
- ResNets

#### Visualizing CNNs
- Visualizing CNN features
- Visualizing responses
- Visualizing learned structures

#### Residual Networks (Lecture 17)
- Detailed analysis
- ResNets as ensembles of shallow networks

#### Applications
- Object detection
- Semantic segmentation
- Face identification
### `Recurrent Neural Networks`
#### Word Embeddings (Lecture 18)
- Neuroprobabilistic Language Models
- word2vec
- GloVe
- Hierarchical Softmax

#### Embeddings in Vision
- Siamese networks
- Triplet loss networks

#### Outlook: Recurrent Neural Networks
#### RNNs (Lecture 19)
- Motivation
- Intuition

#### Learning with RNNs
- Formalization
- Comparison of Feedforward and Recurrent networks
- ***Backpropagation through Time (BPTT)***

#### Problems with RNN Training
- ***Vanishing Gradients***
- Exploding Gradients
- Gradient Clipping

#### Improved hidden units for RNNS
- Long Short -Term Memory (LSTM)
- Gated Recurrent Units (GRU)

#### Applications of RNNs