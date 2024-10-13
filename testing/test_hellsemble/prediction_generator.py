from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from typing import Union


class PredictionGenerator(ABC):
    """
    Abstract class defining the interface for generating predictions.
    It provides the basic structure for generating predictions using a
    provided estimator.
    """

    @abstractmethod
    def make_prediction_train(
        self, estimator: ClassifierMixin, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Generates prediction based on provided estimator and data. Note that
        during execution of these functions some parameters of prediction
        e. g. threshold can be learned and reused during make_prediction.

        Args:
            estimator (ClassifierMixin): Estimator which produces predictions
                for further prediction logic.
                X (Union[pd.DataFrame, np.ndarray]): Data to predict on.

        Returns:
            np.ndarray: Output predictions.
        """
        pass

    def make_prediction(
        self, estimator: ClassifierMixin, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Generates prediction in the evaluation phase. In this method, no
        prediction behavior should be learned. By default it uses
        make_prediction_train. However, it can be overridden by custom logic.

        Args:
            estimator (ClassifierMixin): Estimator which produces predictions
                for further prediction logic.
                X (Union[pd.DataFrame, np.ndarray]): Data to predict.

        Returns:
            np.ndarray: Output predictions.
        """
        return self.make_prediction_train(estimator, X)


class FixedThresholdPredictionGenerator(PredictionGenerator):
    """
    PredictionGenerator that uses a fixed threshold on predicted probabilities.
    It generates predictions by applying a fixed threshold
    to the predicted probabilities obtained from an estimator.
    The estimator must have a `predict_proba` method that returns
    probability estimates. It only support binary classification.

    Attributes:
        threshold (float): The threshold value used for classification.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def make_prediction_train(
        self, estimator: ClassifierMixin, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        assert hasattr(
            estimator, "predict_proba"
        ), "In this generator estimator must be able to return probabilities!"
        proba = estimator.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
