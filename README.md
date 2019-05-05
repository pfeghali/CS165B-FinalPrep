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
There are 4 main types of ML. These categories define the type of models and algorithims used to trian.

## Supervised
Supervised Learning is when the input data is well deined and lebelled. The input is mapped to a correct corresponding output, and there are no unknnown variables in the data. This is usually the most easy to work with, with high levels of complexity when dealing with hard problems. Particularly, you may not have enough data, which is where semi-supervised learning comes in.

## Semi-Supervised
Semi-Supervised learning takes place when a user has a set of data that is labelled, and a (usually) much larger set of data that is unlabelled. Lets imagine the problem of mapping ocean floor. We may have limited data that we've seen and labelled, but we have massive tranches of data that may simply be too much data to label. This is an extremely commonn problem.

## Unsupervised
Unsupervised learning is when a user has a set of data points with correlation, but no necessary labelling. My favorite example is word2vec. Imagine a data set full of news articles in French, and you were told to explain what 'autre' means, but in connotation, not simply in definition. Obviously connotation is not intrinsically in the definition of a word, but comes from its context, and is nearly impossible to derive in a vaccum. Word2vec looks at the context of words to derive a symantic meaning, which is then mapped to a vector space. This aloows for the comparison of different words and their meaning as vectors, and is a new way of lookng at language that would have been nearly impossible with simply labelled datasets.

## Reinforcement learning
Reinforcement learning is the utilization of experience to learn. An engineer allows a model to reck havoc in some sort of enviornment, just knowing its output space and a set of inputs. It then has to get a reward for completing a task. The agent has no knowledge of what the task is beforehand, and must learn it from scratch by improving its reward. This method of learning is solely through exploration, and has achieved superhuman performance at a variety of problems. It is computationally expensive, but unbelievably effective.

# Key ML Tasks
Machine Learning tasks can be put into a few categories.

## Predective vs Descriptive
Models can generally be split into two categories, predictive or descruptive models. Predective models are characterized by whether or not they have an output variable which contains the target variable. If the model output contains the target variable in some sense, then it is considered a predictive model, and otherwise it is considered a descriptive model.

## Tasks:
Tasks are the sort of things we can make ML models do. What is the difference between a model that classifies fruits and vegetables, and a model that classifies rocks and potatos? Other than the weights and paramaters and such, they are both fundementally a binary classification task.
Due to inherent forms of problems, we can disseminate the main type of tasks that ML models will need to accomplish.

### Classification
Classification tasks are when a model is asked to classify an input as some pre-defined output class. [If there are two potential outputs, then the problem can be considered binary classification.](https://www.youtube.com/watch?v=ACmydtFDTGs) This can be generalized to K-Class Classification, where an input can be mapped to K classes.

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

# Basic Linear Classifer

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