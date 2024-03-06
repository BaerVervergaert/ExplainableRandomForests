# Introduction

This package provides an explainability tool to use in analysis of random forest models.

The core functions provided are:
- a data similarity score
- a data extrapolation score
- a data quantity score

# Explanation

All scores are based on the topology induced by the underlying decision trees of the random forest models.

The data similarity score informs on how similar a data sample is to a reference point (or several reference points) based on how often these end in the same leaf of the decision tree.

The data extrapolation score informs on how much a data sample is extrapolating by tracking the distance to the decision bounds it encountered. We can also include the bounds of the training samples which also ended in the same leaves to further specify the extrapolation score. Unused features are ignored in the score.

The data quantity score informs on how many training samples were used to train the leaves of the tree. There are two variations on this value. Because of various training strategies not all training samples need be consumed during training. We have two different strategies, 'seen' and 'reweighted', which count either only the samples that were used to train the leaf values, or the proportion of the total input data that was used.

All scores are first calculated on decision trees and then aggregated over the trees by calculating the weighted means. The weights of the mean are determined by the weights of the ensemble.

# Examples

First we generate some data and train a random forest.
```python
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def setup_data():
    N = 10_000
    N_train = int(.9*N)
    f = 3
    time = np.arange(N)
    X = np.random.normal(size=(N,f))*.05 + np.sin((2*np.pi*time/(N/2))[:,None]+np.linspace(0,np.pi,f)[None,:])
    y = np.random.normal(size=(N)).cumsum()*.05 + np.sin(2*np.pi*time/(N/2))
    test_extrap_noise = np.random.normal(size=(N-N_train,f)).cumsum(axis=0)*.0
    return X[:N_train], y[:N_train], X[N_train:]+test_extrap_noise, y[N_train:]

def setup_sklearn_random_forest():
    X, y, X_test, y_test = setup_data()
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=3)
    model.fit(X,y)
    return model, X, y, X_test, y_test

model, X, y, X_test, y_test = setup_sklearn_random_forest()
y_all = np.concatenate([y,y_test])
total_X = np.concatenate([X,X_test])

print(predict_train := model.predict(X))
print(predict_test := model.predict(X_test))
predict = np.concatenate([predict_train,predict_test])

# INPUT DATA
for i in range(total_X.shape[1]):
    plt.plot(np.arange(len(total_X[:,i])),total_X[:,i],'.')
plt.axvline(X.shape[0])
plt.show()

# OUTPUT DATA
plt.plot(np.arange(len(y_all)),y_all,'.')
plt.plot(np.arange(len(predict)),predict,'.')
plt.axvline(y.shape[0])
plt.show()

```

Then we create random forest explainer.
```python
from ExplainableRandomForests.Forests.forests import SklearnRandomForestExplainer

model_explainer = SklearnRandomForestExplainer(model)
```

We can now compute the data similarity score. This is answering the question: Which points in X_test and X are similar to the last two points of X_test?
```python
print(similarity_test := model_explainer.data_similarity(X_test, X_test[-2:]))
print(similarity_train := model_explainer.data_similarity(X, X_test[-2:]))
similarity = np.concatenate([similarity_train,similarity_test])

# DATA SIMILARITY
sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=similarity, palette=sns.color_palette('flare', as_cmap=True))
plt.plot(np.arange(len(predict)), predict, c='k', zorder=-1, alpha=.3)
plt.axvline(len(y))
plt.title('Data similarity')
plt.show()
```

We can now compute the data similarity score. This is answering the question: Which points in X_test and X are similar to the last two points of X_test?
```python

print(data_extrapolation_test := model_explainer.data_extrapolation(X_test, strategy='naive'))
print(data_extrapolation_train := model_explainer.data_extrapolation(X, strategy='naive'))
data_extrapolation = np.concatenate([data_extrapolation_train,data_extrapolation_test])

# DATA EXTRAPOLATION
sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_extrapolation.mean(axis=1), palette=sns.color_palette('flare', as_cmap=True))
plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
plt.axvline(len(y))
plt.title('Data extrapolation')
plt.show()
```

We can now compute the data similarity score. This is answering the question: Which points in X_test and X are similar to the last two points of X_test?
```python
print(data_extrapolation_w_train_test := model_explainer.data_extrapolation(X_test, train_X=X, strategy='include_training_data'))
print(data_extrapolation_w_train_train := model_explainer.data_extrapolation(X, train_X=X, strategy='include_training_data'))
data_extrapolation_w_train = np.concatenate([data_extrapolation_w_train_train,data_extrapolation_w_train_test])

# DATA EXTRAPOLATION W TRAIN
sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_extrapolation_w_train.mean(axis=1), palette=sns.color_palette('flare', as_cmap=True))
plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
plt.axvline(len(y))
plt.title('Data extrapolation with train data')
plt.show()
```

We can now compute the data similarity score. This is answering the question: Which points in X_test and X are similar to the last two points of X_test?
```python
print(data_train_observations_test := model_explainer.data_train_observations(X_test, strategy='seen'))
print(data_train_observations_train := model_explainer.data_train_observations(X, strategy='seen'))
data_train_observations = np.concatenate([data_train_observations_train,data_train_observations_test])

# DATA OBSERVATIONS
sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_train_observations, palette=sns.color_palette('flare', as_cmap=True))
plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
plt.axvline(len(y))
plt.title('Observations used during train')
plt.show()
```

We can now compute the data similarity score. This is answering the question: Which points in X_test and X are similar to the last two points of X_test?
```python
print(data_train_observations_weighted_test := model_explainer.data_train_observations(X_test, strategy='weighted'))
print(data_train_observations_weighted_train := model_explainer.data_train_observations(X, strategy='weighted'))
data_train_observations_weighted = np.concatenate([data_train_observations_weighted_train,data_train_observations_weighted_test])

# DATA OBSERVATIONS WEIGHTED
sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_train_observations_weighted, palette=sns.color_palette('flare', as_cmap=True))
plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
plt.axvline(len(y))
plt.title('Observations used during train reweighted')
plt.show()
```




# Installation

Package not yet available on PyPi
