---
Peter Feghali @ Spring 2019
UCSB `CS165B` Final Study Guide
---
#### Introduction:
This is an incomplete final study guide.
There is probably material missing from here, albeit most information should be valid.
Do not expect substantive proofs, that's not what this is for. This will be continously updated as the course progresses.
Please email me if you'd like something updated or changed.
---
##### *All code blocks are mine unless otherwise noted, and might have errors/be suboptimal. I will be using pseudocode. Other documents and references will be linked to. Sorry about the typos.*
---

# What is ML
Defined in the lecture as: "Machine learning is the design and analysis of algorithms that improve their performance at some task with experience".
ML is defined not only by the algoithims, but particulary the power of the applications it has. We use the term learning quite loosely here.

## Tasks
We define a task as a peice of work that the algorithm is designed to complete. The task must be well defined before working on any machine learning problem, as your task definition defines the type of model and feauture engineering required.
Some examples of tasks are classification, predicting a given output via a set of input data, etc. You can think about a task in the same way you as a human would view a task - What am I being asked to do?

## Experience
Experience is the set of data that you use to build your model. This data can be in any form, and should optimally be highly correlated to your expected outpt, or have enough intrinisc information to properly map to an output when multiple sources are combined. It is important to recognize "Garbage In -> Garbage Out" - Having good input data is a necessity to having a good model. Throwing a bunch of bad data to a neural network may give you an output, but truly, if your data isn't good you should reaxmine your problem domain and find a better way to work with your task at hand.

## Features
Sometimes your input data might seem like garbage, until you work with the data. Different than data 'massaging' to get an optimal output, features are the set of inputs that you give to your Machine Learning model. Sometimes these are the same as your orginal data points, but they do not have to be.

### Feature Engineering
Feature engineering is the process of applying proceeses to data to extract useful information and disseminate what matters from a dataset, to then feed into a model. While it is true, that (large) models could (and should) theoretically learn feature engineering on their own given enough time, there is almost always no good reason to not conduct the feature engineering beforehand, and optimize the learning process.

## Performance
Performance is determining how effective ones model. After training (or during, to continously evluate overall performance), models or tested on a set of previously unseen testing data, to verify whether or not the model actually learned anything. This set of testing data is usually a randomly chosen set of the testing data, held out from training. This is integral to achieving high performing models, as verifying performance with a seperate dataset allows for the mitigation of error and overfitting.

## Models
Models are the tools that we use to conduct machine learning. Choosing the right model and knowing which to use in what use case, is a fundemental skill. Tools such as AutoML help optimize this process, but as ML engineers, understanding a problem and model behavior rather than just throwing raw computing power against a problem is important. Disseminating when regression should be used or a decision tree, is important. While both may be effective for the problem you're trying to solve, each has inherit advantages and disadvantages.

## Online vs Offline Learning
There are two overarching types of learning, online and offline learning. Offline learning is significantly more common, which is utilizing a set of predetermined data points to train a model, then verifying performance, and applying it to a real problem. This model is then fixed with defined weights, and is usually never retrained. Online learning is the process of taking a ML model and putting it into the world, then having it make decesions while evaulating performance live. This allows for the continous learning of a problem with an increasing array of data, but may not be easy to work with.

# Types of ML

## Supervised

## Semi-Supervised

## Unsupervised

## Reinforcement learning

# Key ML Tasks

## Predective vs Descriptive

## Tasks:

### Classification

### Clustering

### Regression

### Dimensionality Reduction

### Anolmaly Detection

# Types of ML models

## Geometric Models

## Probablistic Models

## Logical Models

# Generalization vs Overfitting

# Intrinsic Dimensions

# Inductive Bias

# The Curse of Dimensionality

# Norms (Distance Measures)

# Contingency Plots

### True Positive

### False Positives

### False Negatives

### False Positives

### Accuracy

### Precision

### Error Rate

### Recall

### Sensitivity

### F1 Score

# Accuracy vs Precision vs Recall

## Deciding what you want and why

## Deciding on how to optimize for that objective

# Coverage plots / ROC Plot

# Scoring Classifiers

## Margin

## Loss Functions

### Minimizing Overfitting with intelligent loss function choices

# Ranking Classifiers

## Error Assesment

## Coverage Curves

# LaPlace correction

# Empirical probability

# M-Estimate

# MAP Decision rule

# Least Squares Model

# Regression Models

## Residual Error Optimization

## Multivariate Linear Regression

## Linear Least Squares Regression

### Univariate

### Multivariate

### Least Squares Classifier

# Trees

## Decision Trees vs Feature Trees

## Decision Trees

## Feature Trees

## Ranking Trees

## Probability Estimation Trees

## Training Trees

### Best Split algo

# Stats

### Mean

### Variance

### Covariance

### Covariance Matrix

### Uncorrelated Variables

## Bayes Rule

# Math + More

## Homogenous vs Non-Homogenous Coordinate Representations

## Outliers

## Regularization

# The Perceptron Model