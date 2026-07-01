# Machine Learning Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive collection of machine learning algorithms implemented from scratch in Python, accompanied by detailed experimental analysis. This repository contains Jupyter notebooks covering fundamental concepts in statistical learning theory, linear models, decision trees, support vector machines, and deep learning architectures including Neural Networks, CNNs, and RNNs. Each implementation is validated through rigorous experiments on synthetic and real-world datasets, with results documented in the accompanying lab reports.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Lab 1: PAC Learnability and Regression](#lab-1-pac-learnability-and-regression)
  - [Task 1: Threshold Hypotheses](#task-1-threshold-hypotheses)
  - [Task 2: Polynomial Regression](#task-2-polynomial-regression)
  - [Task 3: Rectangular Hypotheses](#task-3-rectangular-hypotheses)
- [Lab 2: Classification Models](#lab-2-classification-models)
  - [Task 1: Logistic Regression Variants](#task-1-logistic-regression-variants)
  - [Task 2: Decision Trees](#task-2-decision-trees)
  - [Task 3: Support Vector Machines](#task-3-support-vector-machines)
- [Lab 3: Neural Networks, CNN, and RNN](#lab-3-neural-networks-cnn-and-rnn)
  - [Task 1: Feedforward Neural Networks](#task-1-feedforward-neural-networks)
  - [Task 2: Convolutional Neural Networks](#task-2-convolutional-neural-networks)
  - [Task 3: Recurrent Neural Networks](#task-3-recurrent-neural-networks)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Key Findings](#results-and-key-findings)
- [License](#license)
- [Author](#author)

---

## Overview

This repository demonstrates the theoretical foundations and practical implementations of core machine learning algorithms, progressively building from classical statistical learning theory up to modern deep learning architectures. The code is written from scratch, relying primarily on NumPy for numerical operations and Matplotlib/Seaborn for visualisation, with PyTorch used for deep learning experiments in Lab 3. The accompanying lab reports (PDF) provide detailed mathematical derivations, experimental methodology, and analysis of results.

**Key Features:**

- Pure Python/NumPy implementations of classical ML algorithms
- PyTorch-based deep learning implementations (Neural Networks, CNN, RNN)
- Comprehensive experimental analysis of bias-variance tradeoffs
- Study of sample complexity and generalisation bounds
- Comparison of regularisation techniques (L1, L2, Dropout, Batch Normalisation)
- Analysis of kernel methods and hyperparameter effects
- End-to-end deep learning pipelines for image classification and sequence modelling
- Clean, well-documented code with visualisation utilities

---

## Repository Structure

```
Machine-Learning-Algorithms/
├── Regression & PAC Bounds/          # Lab 1 notebooks and data
├── Classification Models/            # Lab 2 notebooks and data
├── Neural Networks, CNN, RNN/        # Lab 3 notebooks and data
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Lab 1: PAC Learnability and Regression

### Task 1: Threshold Hypotheses

Implementation of one-dimensional threshold classifiers under the PAC learning framework.

- **Data Generation**: Two Gaussian distributions (class 1: N(3,1), class 0: N(4.5,1))
- **Empirical Risk Minimisation (ERM)**: Search over midpoints between data points
- **Optimal Threshold**: Analytical derivation shows Bayes optimal threshold at 3.75 with irreducible error 22.66%
- **Sample Complexity**: Empirical analysis of convergence to Bayes error, demonstrating that 80–100 samples achieve epsilon = 0.01 accuracy
- **Theoretical Analysis**: VC dimension of threshold functions (VCdim = 2) and corresponding sample complexity bounds

### Task 2: Polynomial Regression

Comprehensive analysis of polynomial regression using both analytical and stochastic gradient descent methods.

- **Analytical Solution**: Normal equations with closed-form weight computation
- **SGD Implementation**: Minibatch gradient descent with feature standardisation
- **Bias-Variance Tradeoff**: Investigation of polynomial degrees M in {2, 4, 5, 7, 10, 15}
- **Regularisation**: Ridge (L2) and Lasso (L1) regression with cross-validated lambda selection
- **Runtime Analysis**: Scaling behaviour with sample size n and polynomial degree M
- **Sample Complexity**: Empirical determination of minimum n to achieve MSE < 0.05 for varying M
- **Non-Realisable Setting**: Analysis with Poisson noise, demonstrating convergence to minimum achievable risk

### Task 3: Rectangular Hypotheses

PAC learning of axis-aligned rectangle classifiers in two dimensions.

- **Hypothesis Class**: Rectangles [0, a] x [b, 1] on the unit square
- **ERM Implementation**: Search over O(n²) data-induced thresholds
- **Convergence Analysis**: True risk vs. sample size for n in {10, 20, 50, 100, 200, 500, 1000}
- **Sample Complexity Bounds**: Comparison of theoretical VC bounds (n ~ 2000) with empirical requirements (n ~ 200)
- **Agnostic Setting**: Learning with 20% label noise; analysis of irreducible error and consistency guarantees

---

## Lab 2: Classification Models

### Task 1: Logistic Regression Variants

Implementation and comparison of multiple logistic regression architectures on the Car Evaluation dataset (UCI).

- **Linear Classifier (Multiclass Perceptron)**: Batch update rule with 90.22% training accuracy
- **One-vs-Rest (OvR) Logistic Regression**: Three binary classifiers with sigmoid output; 88.73% test accuracy
- **Regularisation**: L1 (Lasso) and L2 (Ridge) penalties; L1 achieves 90.17% test accuracy with sparse feature selection
- **Ordinal Logistic Regression**: Cumulative logit model respecting class order; achieves 94.80% test accuracy with zero extreme errors
- **Comparative Analysis**: Detailed confusion matrices and Mean Absolute Error (MAE) evaluation

### Task 2: Decision Trees

From-scratch implementation of decision tree classifiers with extensive analysis of splitting criteria and ensemble methods.

- **Splitting Criteria**: Entropy (information gain) and Gini impurity
- **Tree Depth Analysis**: Bias-variance tradeoff for max_depth in [1, 10]; optimal depth found to be 3
- **Random Forest**: Implementation of bagging with random feature selection; variance reduction demonstration
- **Evaluation**: Precision, recall, F1-score, and confusion matrices on the Iris dataset

### Task 3: Support Vector Machines

One-vs-One (OVO) SVM implementation with linear and kernel variants.

- **Linear SVM**: Primal formulation with hinge loss and subgradient descent
- **Hyperparameter Tuning**: Cross-validation for C in {0.01, 0.1, 1.0, 10.0, 100.0}; optimal C = 0.1 yields 93.33% test accuracy
- **Kernel SVM**: Polynomial (degree 2) and RBF kernels implemented via dual formulation
- **Decision Boundary Visualisation**: 2D plots showing linear and non-linear separations
- **Support Vector Analysis**: Effect of C on number of support vectors and margin violations

---

## Lab 3: Neural Networks, CNN, and RNN

### Task 1: Feedforward Neural Networks

From-scratch and PyTorch implementations of multi-layer perceptrons for classification and regression.

- **Architecture Design**: Fully connected networks with configurable hidden layers, neurons, and activation functions (ReLU, Sigmoid, Tanh)
- **Backpropagation**: Manual derivation and implementation of gradient computation via chain rule
- **Optimisers**: Comparison of SGD, SGD with momentum, and Adam; analysis of convergence speed and stability
- **Regularisation**: Dropout and L2 weight decay; impact on overfitting and generalisation
- **Batch Normalisation**: Effect on training stability and convergence rate
- **Hyperparameter Study**: Learning rate schedules, batch size effects, and network depth vs. width tradeoffs

### Task 2: Convolutional Neural Networks

Implementation of CNN architectures for image classification tasks.

- **Convolutional Layers**: Manual implementation of 2D convolution, padding, and stride; learned filter visualisation
- **Pooling Operations**: Max pooling and average pooling; effect on spatial dimensionality and translation invariance
- **Architecture Variants**: Shallow custom CNNs vs. deeper LeNet-style architectures; analysis of receptive fields
- **Data Augmentation**: Random cropping, horizontal flipping, and normalisation to improve generalisation
- **Training Pipeline**: End-to-end training on image datasets with learning curves, accuracy/loss tracking, and confusion matrices
- **Feature Map Visualisation**: Intermediate activation maps showing hierarchical feature extraction

### Task 3: Recurrent Neural Networks

Implementation of RNN and LSTM architectures for sequence modelling tasks.

- **Vanilla RNN**: Manual implementation of recurrent hidden states; analysis of vanishing gradient problem across long sequences
- **Sequence Tasks**: Time-series prediction and text generation experiments demonstrating temporal modelling capability
- **Gradient Analysis**: Visualisation of gradient norms across time steps; empirical confirmation of vanishing/exploding gradients
- **Hyperparameter Study**: Hidden size, number of layers, sequence length, and teacher forcing ratio
- **Evaluation Metrics**: Perplexity for language modelling, MSE for regression tasks

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/arjunaggarwaliit/Machine-Learning-Algorithms.git
cd Machine-Learning-Algorithms
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Navigate to the desired lab directory and launch Jupyter:

```bash
jupyter notebook
```

Open the corresponding notebook:

- **`Regression & PAC Bounds/Lab1_PAC_Learnability_Regression.ipynb`** — Experiments on threshold functions, polynomial regression, and rectangular hypotheses.
- **`Classification Models/Lab2_Classification_Models.ipynb`** — Implementation of logistic regression variants, decision trees, and SVMs.
- **`Neural Networks, CNN, RNN/Lab3_Neural_Networks_CNN_RNN.ipynb`** — Implementation of feedforward networks, CNNs, and RNNs with deep learning experiments.

- Each notebook contains executable cells with detailed comments and visualisation code. Results can be reproduced by running all cells sequentially.
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Arjun Aggarwal**
Roll No: 2024AIB1289
GitHub: [@arjunaggarwaliit](https://github.com/arjunaggarwaliit)

For questions or collaboration, please open an issue on the repository.
