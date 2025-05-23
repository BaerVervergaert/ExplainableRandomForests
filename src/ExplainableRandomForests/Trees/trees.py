import numpy as np


class BoundingBox:
    def __init__(self,dim: int,lower: np.ndarray, upper:np.ndarray):
        r"""BoundingBox is a class that represents a bounding box in n-dimensional space.

        Parameters
        ----------
        dim : int
            The number of dimensions of the bounding box.
        lower : np.ndarray
            A 1D array of length dim representing the lower bounds of the bounding box.
        upper : np.ndarray
            A 1D array of length dim representing the upper bounds of the bounding box.
        """
        self._validate_dimension_length(dim, lower, upper)
        self._validate_lower_smaller_eq_then_upper(lower, upper)
        self.dim = dim
        self.lower = lower
        self.upper = upper

    def _validate_lower_smaller_eq_then_upper(self, lower, upper):
        if (lower > upper).any():
            raise ValueError("Array lower is not always smaller than upper.")

    def _validate_dimension_length(self, dim, lower, upper):
        if len(lower) != dim or len(upper) != dim:
            raise ValueError(f'Expected lower and upper to be of length {dim}, but got: {len(lower)} and {len(upper)}.')

    def apply(self, X:np.ndarray):
        r"""Apply the bounding box as an indicator function to a set of points in n-dimensional space.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points to be checked.
        """
        self._validate_dimension_input(X)
        return np.apply_along_axis(lambda x, self=self: x in self, -1, X)

    def __contains__(self, item):
        return (self.lower<item).all()&(item<=self.upper).all()

    def _validate_dimension_input(self, X):
        if X.shape[-1] != self.dim:
            raise ValueError(f"Expected input to have last dimension of size {self.dim}, but found shape: {X.shape}")

    @classmethod
    def gen_from_bounds(cls,dim,bounds):
        r"""Generate a bounding box from a list of bounds.

        Parameters
        ----------
        dim : int
            The number of dimensions of the bounding box.
        bounds : list of tuples
            A list of tuples, where each tuple contains the feature index, bound type ('lower' or 'upper'), and the bound value.
        """
        lower = np.ones(dim)*-np.inf
        upper = np.ones(dim)*np.inf
        for feature, bound_type, bound in bounds:
            if bound_type.lower().strip() == 'lower':
                lower[feature] = max(lower[feature],bound)
            elif bound_type.lower().strip() == 'upper':
                upper[feature] = min(upper[feature],bound)
        return cls(dim,lower,upper)


class DecisionTreeExplainer:
    def __init__(self,model):
        r"""DecisionTreeExplainer is a class that provides an interface to explain the decision tree model.

        Parameters
        ----------
        model : DecisionTreeRegressor or DecisionTreeClassifier
            A decision tree model from sklearn.
        """
        self.model = model
    @property
    def tree(self):
        r"""Get the tree attribute of the model."""
        return self.model.tree_
    @property
    def threshold(self):
        """Get the threshold attribute of the tree."""
        return self.tree.threshold
    @property
    def children(self):
        """Get the children of the tree."""
        return self.tree.children_left, self.tree.children_right
    @property
    def feature(self):
        """Get the feature attribute of the tree."""
        return self.tree.feature
    @property
    def value(self):
        """Get the value attribute of the tree."""
        return self.tree.value
    @property
    def n_features(self):
        """Get the number of features in the tree."""
        return self.tree.n_features
    def neighborhood(self, x):
        r"""Get the neighborhood of a point in the decision tree.

        Parameters
        ----------
        x : np.ndarray
            A 1D array representing the point for which to find the neighborhood.
        Returns
        -------
        BoundingBox
        """
        self._validate_neighborhood_input(x)
        X = x.reshape(1,-1).astype(np.float32)
        path = self.tree.decision_path(X).indices[:-1]
        path_features = self.feature[path]
        path_thresholds = self.threshold[path]
        input_values = x[path_features]
        input_is_lower = input_values <= path_thresholds
        bounds = []
        for i in range(len(path)):
            bound_type = 'upper' if input_is_lower[i] else 'lower'
            bound = (path_features[i],bound_type,path_thresholds[i])
            bounds.append(bound)
        return BoundingBox.gen_from_bounds(self.n_features,bounds)
    def data_neighborhood(self,X,reference):
        r"""Mark in a dataset the neighborhood of a reference point in the decision tree.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the neighborhood.
        reference : np.ndarray
            A 1D array representing the reference point for which to find the neighborhood.
        Returns
        -------
        np.ndarray
            A 1D boolean array indicating whether each point in X is in the neighborhood of the reference point.
        """
        box = self.neighborhood(reference)
        return box.apply(X)
    def data_neighborhood_multi_reference(self,X,references):
        """Mark in a dataset the neighborhood of multiple reference points in the decision tree.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the neighborhood.
        references : np.ndarray
            A 2D array of shape (n_references, n_features) representing the reference points for which to find the neighborhood.
        Returns
        -------
        np.ndarray
            A 2D boolean array indicating whether each point in X is in the neighborhood of each reference point.
        """
        masks = []
        for reference in references:
            ref_mask = self.data_neighborhood(X,reference)
            masks.append(ref_mask)
        return np.stack(masks,axis=-1)
    def _validate_neighborhood_input(self, x):
        """Validate the input for the neighborhood function."""
        if not len(x.shape) == 1:
            raise ValueError(f"Expected an array. Got shape: {x.shape}")
        if len(x) != self.n_features:
            raise ValueError(f"Expected array to have dimension: {self.n_features}, but got {len(x)}")
    def multi_bounds(self,X):
        """Calculate the bounds for each feature in the dataset.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the bounds.
        Returns
        -------
        left_bound : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the lower bounds for each feature.
        right_bound : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the upper bounds for each feature.
        """
        dtr = self.model
        tree = dtr.tree_
        decision_path = dtr.decision_path(X)
        left_bound = np.ones(X.shape)*-np.inf
        right_bound = np.ones(X.shape)*np.inf
        for i in range(tree.n_features):
            feature_mask = tree.feature == i
            if feature_mask.any():
                X_column = X[:, i]
                thresholds = tree.threshold[feature_mask]
                smaller_mask = X_column[:, None] <= thresholds[None, :]
                visited = decision_path[:, feature_mask].todense()
                right_bound[:, i] = np.where(np.multiply(smaller_mask, visited), thresholds[None, :], np.inf).min(axis=1)
                left_bound[:, i] = np.where(np.multiply(~smaller_mask, visited), thresholds[None, :], -np.inf).max(axis=1)
        return left_bound, right_bound
    def data_bounds(self,X):
        """Calculate the bounds for each feature in the dataset.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the bounds.
        Returns
        -------
        left_bound : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the lower bounds for each feature.
        right_bound : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the upper bounds for each feature.
        """
        dtr = self.model
        tree = dtr.tree_
        classes = dtr.apply(X)
        n_features = tree.n_features
        n_leaves = tree.node_count
        left_bound = np.ones((n_leaves,n_features))*np.inf
        right_bound = np.ones((n_leaves,n_features))*-np.inf
        for i in range(n_leaves):
            mask_classes = classes == i
            if mask_classes.any():
                left_bound[i,:] = X[mask_classes,:].min(axis=0)
                right_bound[i,:] = X[mask_classes,:].max(axis=0)
        return left_bound, right_bound
    def sample_data_bounds(self,train_X, X):
        """Calculate the bounds for each feature in the dataset.

        Parameters
        ----------
        train_X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the training points for which to find the bounds.
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the bounds.
        Returns
        -------
        left_bound : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the lower bounds for each feature.
        right_bound : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the upper bounds for each feature.
        """
        dtr = self.model
        left_bound, right_bound = self.data_bounds(train_X)
        classes = dtr.apply(X)
        return left_bound[classes,:], right_bound[classes,:]
    def data_extrapolation_naive(self,X):
        """Calculate the naive extrapolation for each feature in the dataset.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the extrapolation.
        Returns
        -------
        extrapolation_value : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the extrapolation value for each feature.
        """
        left_bound, right_bound = self.multi_bounds(X)
        left_extrapolate = np.isinf(left_bound)
        right_extrapolate = np.isinf(right_bound)
        exclude = left_extrapolate & right_extrapolate
        soft_left_extrapolate = left_extrapolate*(~exclude)
        soft_right_extrapolate = right_extrapolate*(~exclude)
        left_extrapolation_value = np.where(soft_left_extrapolate,(right_bound - X),0.)
        right_extrapolation_value = np.where(soft_right_extrapolate,(X-left_bound),0.)
        extrapolation_value = np.where(exclude,np.nan,left_extrapolation_value+right_extrapolation_value)
        return extrapolation_value
    def data_extrapolation_with_train(self,train_X, X):
        """Calculate the extrapolation for each feature in the dataset, including training data.

        Parameters
        ----------
        train_X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the training points for which to find the bounds.
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the bounds.
        Returns
        -------
        extrapolation_value : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the extrapolation value for each feature.
        """
        naive_extrapolation_value = self.data_extrapolation_naive(X)
        train_extrapolation_value = self.data_extrapolation_only_train(train_X,X)
        return np.minimum(naive_extrapolation_value,train_extrapolation_value)
    def data_extrapolation_only_train(self,train_X,X):
        """Calculate the extrapolation for each feature in the dataset, using only training data.

        Parameters
        ----------
        train_X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the training points for which to find the bounds.
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the bounds.
        Returns
        -------
        extrapolation_value : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the extrapolation value for each feature.
        """
        left_bound, right_bound = self.sample_data_bounds(train_X, X)
        left_extrapolate = X < left_bound
        right_extrapolate = X > right_bound
        exclude = left_extrapolate & right_extrapolate
        soft_left_extrapolate = left_extrapolate*(~exclude)
        soft_right_extrapolate = right_extrapolate*(~exclude)
        left_extrapolation_value = np.where(soft_left_extrapolate,(left_bound - X),0.)
        right_extrapolation_value = np.where(soft_right_extrapolate,(X-right_bound),0.)
        extrapolation_value = np.where(exclude,np.nan,left_extrapolation_value+right_extrapolation_value)
        return extrapolation_value
    def data_extrapolation_with_train_old(self, train_X, X):
        # TODO: Remove this function
        left_bound, right_bound = self.multi_bounds(X)
        left_extrapolate = np.isinf(left_bound)
        right_extrapolate = np.isinf(right_bound)
        exclude = left_extrapolate & right_extrapolate
        train_left_bound, train_right_bound = self.sample_data_bounds(train_X, X)
        train_left_extrapolate = X < train_left_bound
        train_right_extrapolate = X > train_right_bound
        hard_left_extrapolate = (left_extrapolate|train_left_extrapolate)*(~exclude)
        hard_right_extrapolate = (right_extrapolate|train_right_extrapolate)*(~exclude)
        left_extrapolation_value = np.where(hard_left_extrapolate,(np.minimum(right_bound,train_right_bound) - X),0.)
        right_extrapolation_value = np.where(hard_right_extrapolate,(X-np.maximum(left_bound,train_left_bound)),0.)
        return left_extrapolation_value + right_extrapolation_value
    def data_train_observations(self,X):
        """Calculate the number of training observations for each feature in the dataset.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the extrapolation.
        Returns
        -------
        data_train_observations : np.ndarray
            A 1D array of shape (n_samples,) representing the number of training observations for each point in X.
        """
        dtr = self.model
        tree = dtr.tree_
        node_samples = tree.n_node_samples
        return node_samples[dtr.apply(X)]
    def data_weighted_train_observations(self,X):
        """Calculate the number of training observations for each feature in the dataset.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the points for which to find the extrapolation.
        Returns
        -------
        data_train_observations : np.ndarray
            A 1D array of shape (n_samples,) representing the number of training observations for each point in X.
        """
        dtr = self.model
        tree = dtr.tree_
        node_samples = tree.weighted_n_node_samples
        return node_samples[dtr.apply(X)]









