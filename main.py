import datetime
import time

import src.ExplainableRandomForests.Forests.forests
from src.ExplainableRandomForests.Trees import trees
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


def setup_ngboost_random_forest():
    X, y, X_test, y_test = setup_data()
    from ngboost import NGBRegressor
    model = NGBRegressor()
    model.fit(X,y)
    return model, X, y, X_test, y_test


class TimeCode:
    def __init__(self):
        self.laps = []
    def __enter__(self):
        self.enter_time = time.time()
        self.laps.append(self.enter_time)
    def __exit__(self,*args):
        self.lap()
    def lap(self):
        lap_time = time.time()
        self.print_lap(lap_time)
        self.laps.append(time.time())
    def print_lap(self,lap_time):
        since_start = self.str_format_diff(lap_time - self.enter_time)
        since_last = self.str_format_diff(lap_time - self.laps[-1])
        msg = f"Lap: {since_last}. Total: {since_start}"
        print(msg)
    def str_format_diff(self,diff_time):
        return str(datetime.timedelta(seconds=diff_time))



if __name__=='__main__':
    model, X, y, X_test, y_test = setup_sklearn_random_forest()
    dtr = model.estimators_[0]
    tree = dtr.tree_
    dtr_explainer = trees.DecisionTreeExplainer(dtr)
    dtr_explainer.data_bounds(X)
    dtr_explainer.multi_bounds(X)
    # box = trees.BoundingBox(3,np.array([0,-1,-2]),np.array([1,0,-1]))
    # print(first_apply := box.apply(X))
    # bounds = [(0,'lower',-1),(0,'lower',0),(0,'lower',-2),(1,'lower',-1),(2,'lower',-2),(0,'upper',1),(1,'upper',0),(2,'upper',2),(2,'upper',-1),(2,'upper',1)]
    # box = trees.BoundingBox.gen_from_bounds(3,bounds)
    # print(second_apply := box.apply(X))
    # print((first_apply==second_apply).all())
    # print(dtr_explainer.tree)
    # print(dtr_explainer.feature)
    # print(dtr_explainer.children)
    # print(dtr_explainer.threshold)
    # print(model.estimators_)
    # print(model, X, y)
    # print('Hi!')
    # print(dtr_explainer.neighborhood(X[0]))
    # print(dtr_explainer.data_neighborhood(X,X[0]))
    # print(dtr_explainer.data_neighborhood_multi_reference(X,X[:3]))
    # print("Here!")
    # model_explainer = src.ExplainableRandomForests.Forests.forests.NGBoostRandomForestExplainer(model)
    # print(similarity := model_explainer.data_similarity(X,X[-10:]))
    # print(predict := model.predict(X))
    #
    # sns.relplot(x=np.arange(len(y)),y=y,hue=similarity,palette=sns.color_palette('flare',as_cmap=True))
    # plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
    #
    # # plt.plot(similarity)
    # # plt.plot(y)
    # plt.show()





if __name__=='__main__':
    # model, X, y = setup_sklearn_random_forest()
    model, X, y, X_test, y_test = setup_ngboost_random_forest()
    # dtr_explainer = trees.DecisionTreeExplainer(dtr)
    # box = trees.BoundingBox(3,np.array([0,-1,-2]),np.array([1,0,-1]))
    # print(first_apply := box.apply(X))
    # bounds = [(0,'lower',-1),(0,'lower',0),(0,'lower',-2),(1,'lower',-1),(2,'lower',-2),(0,'upper',1),(1,'upper',0),(2,'upper',2),(2,'upper',-1),(2,'upper',1)]
    # box = trees.BoundingBox.gen_from_bounds(3,bounds)
    # print(second_apply := box.apply(X))
    # print((first_apply==second_apply).all())
    # print(dtr_explainer.tree)
    # print(dtr_explainer.feature)
    # print(dtr_explainer.children)
    # print(dtr_explainer.threshold)
    # print(model.estimators_)
    # print(model, X, y)
    # print('Hi!')
    # print(dtr_explainer.neighborhood(X[0]))
    # print(dtr_explainer.data_neighborhood(X,X[0]))
    # print(dtr_explainer.data_neighborhood_multi_reference(X,X[:3]))
    print("Here!")
    y_all = np.concatenate([y,y_test])

    model_explainer = src.ExplainableRandomForests.Forests.forests.NGBoostRandomForestExplainer(model,parameter_index=0)

    with TimeCode():
        print(similarity_test := model_explainer.data_similarity(X_test, X_test[-2:]))
        print(similarity_train := model_explainer.data_similarity(X, X_test[-2:]))
        similarity = np.concatenate([similarity_train,similarity_test])

    with TimeCode():
        print(predict_train := model.predict(X))
        print(predict_test := model.predict(X_test))
        predict = np.concatenate([predict_train,predict_test])

    with TimeCode():
        print(data_extrapolation_test := model_explainer.data_extrapolation(X_test, strategy='naive'))
        print(data_extrapolation_train := model_explainer.data_extrapolation(X, strategy='naive'))
        data_extrapolation = np.concatenate([data_extrapolation_train,data_extrapolation_test])

    with TimeCode():
        print(data_extrapolation_w_train_test := model_explainer.data_extrapolation(X_test, train_X=X, strategy='include_training_data'))
        print(data_extrapolation_w_train_train := model_explainer.data_extrapolation(X, train_X=X, strategy='include_training_data'))
        data_extrapolation_w_train = np.concatenate([data_extrapolation_w_train_train,data_extrapolation_w_train_test])

    with TimeCode():
        print(data_train_observations_test := model_explainer.data_train_observations(X_test, strategy='seen'))
        print(data_train_observations_train := model_explainer.data_train_observations(X, strategy='seen'))
        data_train_observations = np.concatenate([data_train_observations_train,data_train_observations_test])

    with TimeCode():
        print(data_train_observations_weighted_test := model_explainer.data_train_observations(X_test, strategy='weighted'))
        print(data_train_observations_weighted_train := model_explainer.data_train_observations(X, strategy='weighted'))
        data_train_observations_weighted = np.concatenate([data_train_observations_weighted_train,data_train_observations_weighted_test])

    # INPUT DATA
    total_X = np.concatenate([X,X_test])
    for i in range(total_X.shape[1]):
        plt.plot(np.arange(len(total_X[:,i])),total_X[:,i],'.')
    plt.axvline(X.shape[0])
    plt.show()

    # DATA SIMILARITY
    sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=similarity, palette=sns.color_palette('flare', as_cmap=True))
    plt.plot(np.arange(len(predict)), predict, c='k', zorder=-1, alpha=.3)
    plt.axvline(len(y))
    plt.title('Data similarity')
    plt.show()

    # DATA EXTRAPOLATION
    sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_extrapolation.mean(axis=1), palette=sns.color_palette('flare', as_cmap=True))
    plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
    plt.axvline(len(y))
    plt.title('Data extrapolation')
    plt.show()

    # DATA EXTRAPOLATION W TRAIN
    sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_extrapolation_w_train.mean(axis=1), palette=sns.color_palette('flare', as_cmap=True))
    plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
    plt.axvline(len(y))
    plt.title('Data extrapolation with train data')
    plt.show()

    # DATA OBSERVATIONS
    sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_train_observations, palette=sns.color_palette('flare', as_cmap=True))
    plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
    plt.axvline(len(y))
    plt.title('Observations used during train')
    plt.show()

    # DATA OBSERVATIONS WEIGHTED
    sns.relplot(x=np.arange(len(y_all)), y=y_all, hue=data_train_observations_weighted, palette=sns.color_palette('flare', as_cmap=True))
    plt.plot(np.arange(len(predict)),predict,c='k',zorder=-1,alpha=.3)
    plt.axvline(len(y))
    plt.title('Observations used during train reweighted')
    plt.show()



