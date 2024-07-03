from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


class PredictionPolicy(ABC):

    @abstractmethod
    def make_prediction(
        self, estimator: ClassifierMixin, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray:
        pass


class FixedThresholdPredictionPolicy(PredictionPolicy):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def make_prediction(
        self, estimator: ClassifierMixin, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray:
        assert hasattr(
            estimator, "predict_proba"
        ), "In this policy estimator must be able to return probabilities!"
        proba = estimator.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
