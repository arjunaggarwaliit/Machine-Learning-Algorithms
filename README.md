# Machine Learning Algorithms

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive collection of machine learning algorithms implemented from scratch in Python, accompanied by detailed experimental analysis. This repository contains Jupyter notebooks covering fundamental concepts in statistical learning theory, linear models, decision trees, and support vector machines. Each implementation is validated through rigorous experiments on synthetic and real-world datasets, with results documented in the accompanying lab reports.

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
- [Installation](#installation)
- [Usage](#usage)
- [Results and Key Findings](#results-and-key-findings)
- [License](#license)
- [Author](#author)

---

## Overview

This repository demonstrates the theoretical foundations and practical implementations of core machine learning algorithms. The code is written entirely from scratch, relying only on NumPy for numerical operations and Matplotlib/Seaborn for visualisation. The accompanying lab reports (PDF) provide detailed mathematical derivations, experimental methodology, and analysis of results.

**Key Features:**
- Pure Python/NumPy implementations of algorithms
- Comprehensive experimental analysis of bias-variance tradeoffs
- Study of sample complexity and generalisation bounds
- Comparison of regularisation techniques (L1, L2)
- Analysis of kernel methods and hyperparameter effects
- Clean, well-documented code with visualisation utilities

---


## Lab 1: PAC Learnability and Regression

*Full report: [ML_LAB1.pdf](reports/ML_LAB1.pdf)*

### Task 1: Threshold Hypotheses

Implementation of one-dimensional threshold classifiers under the PAC learning framework.

- **Data Generation**: Two Gaussian distributions (class 1: N(3,1), class 0: N(4.5,1))
- **Empirical Risk Minimisation (ERM)**: Search over midpoints between data points
- **Optimal Threshold**: Analytical derivation shows Bayes optimal threshold at 3.75 with irreducible error 22.66%
- **Sample Complexity**: Empirical analysis of convergence to Bayes error, demonstrating that 80-100 samples achieve epsilon = 0.01 accuracy
- **Theoretical Analysis**: VC dimension of threshold functions (VCdim = 2) and corresponding sample complexity bounds

### Task 2: Polynomial Regression

Comprehensive analysis of polynomial regression using both analytical and stochastic gradient descent methods.

- **Analytical Solution**: Normal equations with closed-form weight computation
- **SGD Implementation**: Minibatch gradient descent with feature standardisation
- **Bias-Variance Tradeoff**: Investigation of polynomial degrees M in {2, 4, 5, 7, 10, 15}
- **Regularisation**: Ridge (L2) and Lasso (L1) regression with cross-validated lambda selection
- **Runtime Analysis**: Scaling behaviour with sample size n and polynomial degree M
- **Sample Complexity**: Empirical determination of minimum n to achieve MSE < 0.05 for varying M
- **Non-Realizable Setting**: Analysis with Poisson noise, demonstrating convergence to minimum achievable risk

### Task 3: Rectangular Hypotheses

PAC learning of axis-aligned rectangle classifiers in two dimensions.

- **Hypothesis Class**: Rectangles [0, a] x [b, 1] on unit square
- **ERM Implementation**: Search over O(n^2) data-induced thresholds
- **Convergence Analysis**: True risk vs. sample size for n in {10, 20, 50, 100, 200, 500, 1000}
- **Sample Complexity Bounds**: Comparison of theoretical VC bounds (n ~ 2000) with empirical requirements (n ~ 200)
- **Agnostic Setting**: Learning with 20% label noise; analysis of irreducible error and consistency guarantees

---

## Lab 2: Classification Models

*Full report: [2024AIB1289_LAB2_Report.pdf](reports/2024AIB1289_LAB2_Report.pdf)*

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

If `requirements.txt` is not present, the following packages are required:

```
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0    # used only for dataset loading and train/test splits
```

---

## Usage

Navigate to the `notebooks/` directory and launch Jupyter:

```bash
jupyter notebook
```

Open either notebook:

- **Lab1_PAC_Learnability_Regression.ipynb**: Experiments on threshold functions, polynomial regression, and rectangular hypotheses.
- **Lab2_Classification_Models.ipynb**: Implementation of logistic regression variants, decision trees, and SVMs.

Each notebook contains executable cells with detailed comments and visualisation code. Results can be reproduced by running all cells sequentially.

---

## Results and Key Findings

- **Sample Complexity**: Practical sample requirements are often 5-10x smaller than worst-case theoretical bounds for well-behaved distributions.
- **Ordinal Regression**: For ordered targets, ordinal logistic regression significantly outperforms standard multiclass approaches (94.80% vs 88.73%).
- **Regularisation**: L1 regularisation provides interpretable feature selection without sacrificing predictive performance.
- **Bias-Variance**: Optimal polynomial degree for the exponential test function was M = 4-5, balancing underfitting and overfitting.
- **Ensemble Methods**: Random forests reduce variance compared to single decision trees, improving stability without increasing bias.
- **Kernel SVMs**: The RBF kernel uses more support vectors than the polynomial kernel, indicating a more complex decision boundary.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Arjun Aggarwal**  
Roll No: 2024AIB1289  
GitHub: [@arjunaggarwaliit](https://github.com/arjunaggarwaliit)

For questions or collaboration, please open an issue on the repository.
