from src.ExplainableRandomForests.Trees.trees import DecisionTreeExplainer
import numpy as np

class BaseRandomForestExplainer:
    def __init__(self,model):
        """
        Base class for Random Forest Explainers.

        Parameters
        ----------
        model : object
            The random forest model to be explained. This should be a fitted model.
        """
        self.model = model
    def _validate_train_data_based_on_strategy(self, strategy, train_X):
        if strategy == 'include_training_data' and (train_X is None):
            raise ValueError("Strategy 'include_training_data' cannot be used without training data provided.")

    def _validate_data_extrapolation_strategy(self, strategy):
        strategies = ['include_training_data','naive']
        if strategy not in strategies:
            raise ValueError(f"Strategy should be one of {strategies}")
    def _validate_data_train_observations(self, strategy):
        strategies = ['seen', 'weighted']
        if strategy not in strategies:
            raise ValueError(f"Strategy should be one of {strategies}")
    def iterate_over_decision_trees(self):
        """
        Iterate over the decision trees in the random forest model.

        Yields
        ------
        decision_tree : object
            A single decision tree from the random forest model.
        """
        raise NotImplemented()
    def tree_weight(self,i):
        raise NotImplemented()
    def data_extrapolation(self, X, train_X = None, strategy='naive'):
        """
        Calculate the data extrapolation for the given data points.


        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data points for which to calculate the data extrapolation.
        train_X : array-like, shape (n_samples, n_features), optional
            The training data points. If None, the training data is not used.
        strategy : str, optional
            The strategy to use for data extrapolation. Options are 'naive' or 'include_training_data'.
            Default is 'naive'.

        Returns
        -------
        score : float
            The data extrapolation score for the given data points.
        """
        self._validate_data_extrapolation_strategy(strategy)
        self._validate_train_data_based_on_strategy(strategy, train_X)
        score = 0.
        count = 0
        for i,decision_tree in enumerate(self.iterate_over_decision_trees()):
            dtree_explainer = DecisionTreeExplainer(decision_tree)
            if strategy=='naive':
                extrapolate = dtree_explainer.data_extrapolation_naive(X)
            elif strategy=='include_training_data':
                extrapolate = dtree_explainer.data_extrapolation_with_train(train_X, X)
            else:
                raise ValueError(f"Strategy should be one of 'naive' or 'include_training_data'")
            nan_mask = np.isnan(extrapolate)
            score += self.tree_weight(i)*np.where(nan_mask,0.,extrapolate)
            count += self.tree_weight(i)
        return score/count
    def data_train_observations(self,X,strategy='seen'):
        """
        Calculate the data train observations for the given data points.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data points for which to calculate the data train observations.
        strategy : str, optional
            The strategy to use for data train observations. Options are 'seen' or 'weighted'.
            Default is 'seen'.

        Returns
        -------
        score : float
            The data train observations score for the given data points.
        """
        self._validate_data_train_observations(strategy)
        score = 0.
        count = 0.
        for i,decision_tree in enumerate(self.iterate_over_decision_trees()):
            dtree_explainer = DecisionTreeExplainer(decision_tree)
            if strategy == 'seen':
                observations = dtree_explainer.data_train_observations(X)
            elif strategy == 'weighted':
                observations = dtree_explainer.data_weighted_train_observations(X)
            else:
                raise ValueError(f"Strategy should be one of 'seen' or 'weighted'")
            score += observations*self.tree_weight(i)
            count += self.tree_weight(i)
        return score/count
    def data_similarity(self,X,references):
        """
        Calculate the data similarity for the given data points.

        This is a measure of how similar the data points are to the reference points.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data points for which to calculate the data similarity.
        references : array-like, shape (n_samples, n_features)
            The reference points to compare against.
        Returns
        -------
        score : float
            The data similarity score for the given data points.
        """
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
        """
        Base class for Random Forest Explainers.

        Parameters
        ----------
        model : object
            The random forest model to be explained. This should be a fitted model.
        """
        super().__init__(model)
    def tree_weight(self,i):
        return 1.
    def iterate_over_decision_trees(self):
        for decision_tree in self.model.estimators_:
            yield decision_tree


class NGBoostRandomForestExplainer(BaseRandomForestExplainer):
    def __init__(self,model,parameter_index):
        """
        Base class for NGBoost Random Forest Explainers.
        Parameters
        ----------
        model : object
            The NGBoost random forest model to be explained. This should be a fitted model.
        parameter_index : int
            The index of the parameter to be explained.
        """
        super().__init__(model)
        self.parameter_index = parameter_index
    @property
    def scalings(self):
        """
        Get the scaling factors for the decision trees in the random forest model.

        Returns
        -------
        scalings : array-like, shape (n_trees,)
        """
        return self.model.scalings
    def iterate_over_decision_trees(self):
        """
        Iterate over the decision trees in the NGBoost random forest model.

        Yields
        ------
        decision_tree : object
            A single decision tree from the NGBoost random forest model.
        """
        for decision_tree in self.model.base_models:
            yield decision_tree[self.parameter_index]
    def tree_weight(self,i):
        """
        Get the weight of the decision tree at index i.
        Parameters
        ----------
        i : int
            The index of the decision tree.
        Returns
        -------
        weight : float
            The weight of the decision tree at index i.
        """
        return self.scalings[i]

