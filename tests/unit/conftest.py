from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ngboost import NGBRegressor
from src.ExplainableRandomForests.Trees.trees import *
from src.ExplainableRandomForests.Forests.forests import *
import pytest


@pytest.fixture
def regression_dataset():
    x = np.arange(0,100*2).reshape((100,2))
    y = np.arange(100)
    yield x,y

@pytest.fixture
def fullX(regression_dataset):
    return regression_dataset[0]

@pytest.fixture
def regression_test_sample():
    x = np.arange(2*100,2*(100+10)).reshape((10,2))
    yield x

@pytest.fixture
def row1X(regression_dataset):
    return regression_dataset[0][:1]

