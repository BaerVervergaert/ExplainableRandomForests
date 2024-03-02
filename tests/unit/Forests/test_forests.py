from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ngboost import NGBRegressor
from src.ExplainableRandomForests.Trees.trees import *
from src.ExplainableRandomForests.Forests.forests import *
import pytest



@pytest.fixture
def decision_tree_regressor(regression_dataset):
    x, y = regression_dataset
    dtr = DecisionTreeRegressor(
        random_state=1,
        max_depth=3,
    )
    dtr.fit(x,y)
    return dtr

@pytest.fixture
def decision_tree_explainer(decision_tree_regressor):
    return DecisionTreeExplainer(decision_tree_regressor)

@pytest.fixture
def sklearn_random_forest(regression_dataset):
    x,y = regression_dataset
    model = RandomForestRegressor(
        max_depth=3,
        random_state=1,
        n_estimators=2,
    )
    model.fit(x, y)
    return model

@pytest.fixture
def sklearn_random_forest_explainer(sklearn_random_forest):
    explainer = SklearnRandomForestExplainer(sklearn_random_forest)
    return explainer

@pytest.fixture
def ngboost_random_forest(regression_dataset):
    x, y = regression_dataset
    base_learner = DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        splitter="best",
        random_state=1,
    )
    model = NGBRegressor(
        random_state=1,
        n_estimators=2,
        Base = base_learner
    )
    model.fit(x,y)
    return model


def print_dtr(decision_tree_regressor, row1X):
    print('Thresholds:',decision_tree_regressor.tree_.threshold)
    print('Features:',decision_tree_regressor.tree_.feature)
    print('Decision path:',decision_tree_regressor.decision_path(row1X).todense())
    print('Row 1 data:', row1X)

def test_print(sklearn_random_forest,ngboost_random_forest,row1X):
    print('\n')
    print('Sklearn RF:', sklearn_random_forest)
    print(sklearn_random_forest.estimators_)
    for i, dtr in enumerate(sklearn_random_forest.estimators_):
        print()
        print('Tree:',i)
        print_dtr(dtr, row1X)

    print('\n')
    print('NGBoost RF:',ngboost_random_forest)
    print(ngboost_random_forest.base_models)
    for i, dtr in enumerate(ngboost_random_forest.base_models):
        print()
        print('Tree',i)
        print_dtr(dtr[0],row1X)
        print_dtr(dtr[1],row1X)


class TestSklearnRandomForestExplainer:
    def test_init(self,sklearn_random_forest):
        # ACT
        srfe = SklearnRandomForestExplainer(sklearn_random_forest)

        # ASSERT
        assert srfe.model == sklearn_random_forest
    def test_data_similarity(self,sklearn_random_forest_explainer , regression_dataset, regression_test_sample):
        # INIT
        X = regression_dataset[0]

        # ACT
        similarity_score = sklearn_random_forest_explainer.data_similarity(X, regression_test_sample)

        # ASSERT
        assert similarity_score.shape == (X.shape[0],)
        assert similarity_score[0] == 0.
        assert similarity_score[-1] == 1.

    def test_data_extrapolation_naive(self,sklearn_random_forest_explainer,row1X,fullX):
        extrap = sklearn_random_forest_explainer.data_extrapolation(row1X,strategy='naive')

        # row1X: 0, 1
        # Tree0 extrap: nan, 25
        # Tree1 extrap: 29, 59
        assert (extrap[0] == np.array([29.,42.])).all()

    def test_data_extrapolation_naive_outside(self,sklearn_random_forest_explainer,regression_test_sample,fullX):
        extrap = sklearn_random_forest_explainer.data_extrapolation(regression_test_sample,train_X=fullX,strategy='naive')

        assert (extrap>0).all()
    def test_data_extrapolation_train(self,sklearn_random_forest_explainer,row1X,fullX):
        extrap = sklearn_random_forest_explainer.data_extrapolation(row1X,train_X=fullX,strategy='include_training_data')
        print(extrap)

        # row1X: 0, 1
        # Tree0 extrap: nan, 25
        # Tree1 extrap: 29, 59
        assert (extrap[0] == np.array([0.,0.])).all()

    def test_data_extrapolation_train_outside(self,sklearn_random_forest_explainer,regression_test_sample,fullX):
        extrap = sklearn_random_forest_explainer.data_extrapolation(regression_test_sample,train_X=fullX,strategy='include_training_data')
        print(extrap)

        assert (extrap>0).all()
    def test_data_train_observations(self,sklearn_random_forest_explainer,regression_test_sample):
        observations = sklearn_random_forest_explainer.data_train_observations(regression_test_sample,strategy='seen')
        print(observations)
    def test_data_train_observations_weighted(self,sklearn_random_forest_explainer,regression_test_sample):
        observations = sklearn_random_forest_explainer.data_train_observations(regression_test_sample,strategy='weighted')
        print(observations)
