from sklearn.tree import DecisionTreeRegressor
from src.ExplainableRandomForests.Trees.trees import *
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

def test_print(decision_tree_regressor,row1X):
    print('\n')
    print('Decision Tree:',decision_tree_regressor)
    print(decision_tree_regressor.tree_)
    print('Thresholds:',decision_tree_regressor.tree_.threshold)
    print('Features:',decision_tree_regressor.tree_.feature)
    print('Decision path:',decision_tree_regressor.decision_path(row1X).todense())
    print('Row 1 data:', row1X)

class TestBoundingBox:
    def __init__(self):
        ...


class TestDecisionTreeExplainer:
    def test_init(self,decision_tree_regressor):
        # ACT
        dte = DecisionTreeExplainer(decision_tree_regressor)

        # ASSERT
        assert dte.model == decision_tree_regressor
    def test_neighborhood(self,decision_tree_explainer, mocker, row1X):
        # ACT
        box = decision_tree_explainer.neighborhood(row1X[0])

        # ASSERT
        assert row1X[0] in box

    def test_neighborhood_all(self,decision_tree_explainer, mocker, fullX):
        # ACT
        for i in range(fullX.shape[0]):
            row_X = fullX[i]
            box = decision_tree_explainer.neighborhood(row_X)

            # ASSERT
            assert row_X in box

    def test_neighborhood_test_inside(self,decision_tree_explainer, mocker, regression_test_sample, fullX):
        # ACT
        box = decision_tree_explainer.neighborhood(fullX[-1])

        for i in range(regression_test_sample.shape[0]):
            row_X = regression_test_sample[i]

            # ASSERT
            assert row_X in box

    def test_neighborhood_test_outside(self,decision_tree_explainer, mocker, regression_test_sample, fullX):
        # ACT
        box = decision_tree_explainer.neighborhood(fullX[0])

        for i in range(regression_test_sample.shape[0]):
            row_X = regression_test_sample[i]

            # ASSERT
            assert row_X not in box

    def test_multi_bounds(self,decision_tree_explainer, mocker, row1X):
        # ACT
        bounds = decision_tree_explainer.multi_bounds(row1X)

        # ASSERT
        assert (row1X >= bounds[0]).all()
        assert (row1X <= bounds[1]).all()

    def test_multi_bounds_all(self,decision_tree_explainer, mocker, fullX):
        # ACT
        bounds = decision_tree_explainer.multi_bounds(fullX)

        # ASSERT
        assert (fullX >= bounds[0]).all()
        assert (fullX <= bounds[1]).all()

    def test_multi_bounds_test(self,decision_tree_explainer, mocker, regression_test_sample):
        # ACT
        bounds = decision_tree_explainer.multi_bounds(regression_test_sample)

        # ASSERT
        assert (np.isinf(bounds[0])|np.isinf(bounds[1])).all()
        assert np.isinf(bounds[1]).all()

    def test_sample_data_bounds(self,decision_tree_explainer,regression_dataset,row1X):
        # INIT
        X = regression_dataset[0]

        # ACT
        bounds = decision_tree_explainer.sample_data_bounds(X,row1X)

        # ASSERT
        assert (row1X >= bounds[0]).all()
        assert (row1X <= bounds[1]).all()

    def test_sample_data_bounds_all(self,decision_tree_explainer,regression_dataset,fullX):
        # INIT
        X = regression_dataset[0]

        # ACT
        bounds = decision_tree_explainer.sample_data_bounds(X,fullX)

        # ASSERT
        assert (fullX >= bounds[0]).all()
        assert (fullX <= bounds[1]).all()

    def test_sample_data_bounds_test(self,decision_tree_explainer,regression_dataset,regression_test_sample):
        # INIT
        X = regression_dataset[0]

        # ACT
        bounds = decision_tree_explainer.sample_data_bounds(X,regression_test_sample)

        # ASSERT
        assert ~( (regression_test_sample <= bounds[1]).all() & (regression_test_sample >= bounds[0]).all())


    def test_data_bounds(self,decision_tree_explainer,regression_dataset):
        # INIT
        X = regression_dataset[0]

        # ACT
        bounds = decision_tree_explainer.data_bounds(X)
        leaf = decision_tree_explainer.model.apply(X)

        # ASSERT
        assert (X >= bounds[0][leaf,:]).all()
        assert (X <= bounds[1][leaf,:]).all()
    def test_data_extrapolation_only_train(self,decision_tree_explainer,fullX,regression_test_sample):
        extrap = decision_tree_explainer.data_extrapolation_only_train(fullX,regression_test_sample)
        classes = decision_tree_explainer.model.apply(regression_test_sample)
        left_bound, right_bound = decision_tree_explainer.data_bounds(fullX)
        print(extrap)
        print(regression_test_sample - right_bound[classes,:])

    def test_train_observations(self, decision_tree_explainer, fullX):
        samples = decision_tree_explainer.data_train_observations(fullX)
        bins = decision_tree_explainer.model.apply(fullX)
        for i in np.unique(bins):
            mask = bins == i
            values = samples[mask]
            assert (values == values[0]).all()
            assert (values == values[0]).sum() <= values[0]

    def test_weighted_train_observations(self, decision_tree_explainer, fullX):
        samples = decision_tree_explainer.data_weighted_train_observations(fullX)
        bins = decision_tree_explainer.model.apply(fullX)
        for i in np.unique(bins):
            mask = bins == i
            values = samples[mask]
            assert (values == values[0]).all()
            assert (values == values[0]).sum() == values[0]


