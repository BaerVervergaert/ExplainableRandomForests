from src.ExplainableRandomForests.Trees.trees import DecisionTreeExplainer
import numpy as np

class BaseRandomForestExplainer:
    def __init__(self,model):
        self.model = model
    def validate_train_data_based_on_strategy(self, strategy, train_X):
        if strategy == 'include_training_data' and (train_X is None):
            raise ValueError("Strategy 'include_training_data' cannot be used without training data provided.")

    def validate_data_extrapolation_strategy(self, strategy):
        strategies = ['include_training_data','naive']
        if strategy not in strategies:
            raise ValueError(f"Strategy should be one of {strategies}")
    def validate_data_train_observations(self, strategy):
        strategies = ['seen', 'weighted']
        if strategy not in strategies:
            raise ValueError(f"Strategy should be one of {strategies}")
    def iterate_over_decision_trees(self):
        raise NotImplemented()
    def tree_weight(self,i):
        raise NotImplemented()
    def data_extrapolation(self, X, train_X = None, strategy='naive'):
        self.validate_data_extrapolation_strategy(strategy)
        self.validate_train_data_based_on_strategy(strategy, train_X)
        score = 0.
        count = 0
        for i,decision_tree in enumerate(self.iterate_over_decision_trees()):
            dtree_explainer = DecisionTreeExplainer(decision_tree)
            if strategy=='naive':
                extrapolate = dtree_explainer.data_extrapolation_naive(X)
            elif strategy=='include_training_data':
                extrapolate = dtree_explainer.data_extrapolation_with_train(train_X, X)
            nan_mask = np.isnan(extrapolate)
            score += self.tree_weight(i)*np.where(nan_mask,0.,extrapolate)
            count += self.tree_weight(i)
        return score/count
    def data_train_observations(self,X,strategy='seen'):
        self.validate_data_train_observations(strategy)
        score = 0.
        count = 0.
        for i,decision_tree in enumerate(self.iterate_over_decision_trees()):
            dtree_explainer = DecisionTreeExplainer(decision_tree)
            if strategy == 'seen':
                observations = dtree_explainer.data_train_observations(X)
            elif strategy == 'weighted':
                observations = dtree_explainer.data_weighted_train_observations(X)
            score += observations*self.tree_weight(i)
            count += self.tree_weight(i)
        return score/count
    def data_similarity(self,X,references):
        if len(references.shape)==1:
            references = references.reshape(1,-1)
        score = 0
        count = 0
        for i,decision_tree in enumerate(self.iterate_over_decision_trees()):
            dtree_explainer = DecisionTreeExplainer(decision_tree)
            mask = dtree_explainer.data_neighborhood_multi_reference(X,references).mean(axis=-1)
            score += mask*self.tree_weight(i)
            count += self.tree_weight(i)
        return score/count


class SklearnRandomForestExplainer(BaseRandomForestExplainer):
    def __init__(self,model):
        super().__init__(model)
    def tree_weight(self,i):
        return 1.
    def iterate_over_decision_trees(self):
        for decision_tree in self.model.estimators_:
            yield decision_tree


class NGBoostRandomForestExplainer(BaseRandomForestExplainer):
    def __init__(self,model,parameter_index):
        super().__init__(model)
        self.parameter_index = parameter_index
    @property
    def scalings(self):
        return self.model.scalings
    def iterate_over_decision_trees(self):
        for decision_tree in self.model.base_models:
            yield decision_tree[self.parameter_index]
    def tree_weight(self,i):
        return self.scalings[i]

