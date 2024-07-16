import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RANDOM_STATE=42
np.random.seed(RANDOM_STATE)

class Data:
    def __init__(self, default : bool = False) -> None:
        self.default = default
        self.X = np.ndarray([])
        self.y = np.ndarray([])
        self.segments_X = []
        self.segments_y = []

    def add_segment(self, n_features : int, n_observations : int, type : str, shuffle : bool = False):
        """
            Add a new segment to the dataset
        """
        case = {
            "linear": self.generate_linear_segment,
            "non-linear": self.generate_non_linear_segment,
            "complex": self.generate_complex_segment,
            "imbalanced": self.generate_imbalanced_segment,
            "high-dimensional": self.generate_high_dimensional_segment
        }
        X, y = case[type](n_observations, n_features)

        self.segments_X.append(X)
        self.segments_y.append(y)

        if len(self.X.shape) == 0: 
            self.X = X
            self.y = y
        else:
            max_features = max(X.shape[1], self.X.shape[1])
            X = self.pad_features(X, max_features)
            self.X = self.pad_features(self.X, max_features)
            self.X = np.vstack((self.X, X))  
            self.y = np.hstack((self.y, y))

        if shuffle:
            from sklearn.utils import shuffle as sk_shuffle  
            self.X, self.y = sk_shuffle(self.X, self.y, random_state=RANDOM_STATE)

    def generate_linear_segment(self, n_obs : int, n_features : int):
        """
            Generate data for linear models 
        """
        X_linear = np.random.randn(n_obs, n_features)
        y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)

        return X_linear, y_linear
    
    def generate_non_linear_segment(self, n_obs : int, n_features : int):
        """
            Generate data for non-linear models (tree-based)
        """
        X_tree = np.random.randn(n_obs, n_features)
        y_tree = (X_tree[:, 0]**2 + X_tree[:, 1]**2 > 1).astype(int)

        return X_tree, y_tree
    
    def generate_complex_segment(self, n_obs : int, n_features : int):
        """
            Generate data for complex models (neural networks)
        """
        X_complex = np.random.randn(n_obs, n_features)
        y_complex = ((X_complex[:, 0] + X_complex[:, 1] * X_complex[:, 2]) > 0).astype(int)

        return X_complex, y_complex
    
    def generate_imbalanced_segment(self, n_obs : int, n_features : int):
        """
            Generate data for imbalanced models
        """
        X_imbalanced = np.random.randn(n_obs, n_features)
        y_imbalanced = np.zeros(n_obs, dtype=int)
        y_imbalanced[:n_obs // 10] = 1  # 10% class 1

        return X_imbalanced, y_imbalanced
    
    def generate_high_dimensional_segment(self, n_obs : int, n_features : int):
        """
            Generate data for high-dimensional models
        """
        X_sparse = np.random.binomial(1, 0.1, (n_obs, n_features)) * np.random.randn(n_obs, n_features)
        y_sparse = (np.sum(X_sparse[:, :10], axis=1) > 0).astype(int)
        return X_sparse, y_sparse
    
    def pad_features(self, X, max_features):
        n_features = X.shape[1]
        if n_features < max_features:
            padding = np.zeros((X.shape[0], max_features - n_features))
            X = np.hstack((X, padding))
        return X
    
    def plot_segments(self):
        """
            Plot the segments, 
            first two features vs y for each segemnt
        """
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        titles = ['Linear Segment', 'Non-linear Segment', 'Complex Segment', 'Imbalanced Segment', 'Noisy Segment', 'High-dimensional Sparse Segment']

        for i, (segment_X, segment_y, title) in enumerate(zip(self.segments_X, self.segments_y, titles)):
            row, col = divmod(i, 2)
            axes[row, col].scatter(segment_X[:, 0], segment_X[:, 1], c=segment_y, cmap='bwr', alpha=0.6)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Feature 1')
            axes[row, col].set_ylabel('Feature 2')
            axes[row, col].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_segments_pca(self):
        """
            Plot the segments, 
            first two PCA components for each segemnt
        """
        pca = PCA(n_components=2)
        pca_segments_X = [pca.fit_transform(segment_X) for segment_X in self.segments_X]

        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        titles = ['Linear Segment', 'Non-linear Segment', 'Complex Segment', 'Imbalanced Segment', 'Noisy Segment', 'High-dimensional Sparse Segment']

        for i, (segment_X, segment_y, title) in enumerate(zip(pca_segments_X, self.segments_y, titles)):
            row, col = divmod(i, 2)
            axes[row, col].scatter(segment_X[:, 0], segment_X[:, 1], c=segment_y, cmap='bwr', alpha=0.6)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('PCA Component 1')
            axes[row, col].set_ylabel('PCA Component 2')
            axes[row, col].grid(True)

        plt.tight_layout()
        plt.show()