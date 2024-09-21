from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


class EstimatorGenerator(ABC):
    """
    This class provides the basic structure for generating a sequence
    of estimators. It requires subclasses to implement
    the `fit_next_estimator` and `has_next` methods.
    """

    @abstractmethod
    def fit_next_estimator(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> ClassifierMixin:
        """
        Abstract method that fits the next estimator based
        on the provided data.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Returns:
            ClassifierMixin: Fitted estimator
        """
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """
        Method that indicates whether there is a next estimator
        in the sequence.

        Returns:
            bool: True if next estimator can be provided. False otherwise
        """
        pass

    @abstractmethod
    def reset_generator(self) -> None:
        """
        Methods that resets the generator.
        """
        pass


class PredefinedEstimatorsGenerator(EstimatorGenerator):
    """
    EstimatorGenerator that iterates over a pre-defined list of estimators.
    This class generates a sequence of estimators from a pre-defined list.
    It fits each estimator in the list sequentially to the provided data.

    Attributes:
        estimators (list[ClassifierMixin]): A list of pre-defined
            scikit-learn classifier objects.
    """

    def __init__(self, estimators: list[ClassifierMixin]):
        self.__estimators = estimators
        self.__proposals_counter = 0

    def fit_next_estimator(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> ClassifierMixin:
        """
        Fits the next estimator in the list to the provided data.
        Raises StopIteration if there are no more estimators in the list.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Raises:
            StopIteration: If there are no more estimators in the list.

        Returns:
            ClassifierMixin: The fitted estimator.
        """
        if self.__proposals_counter == len(self.__estimators):
            raise StopIteration()
        proposed_estimator = self.__estimators[self.__proposals_counter]
        self.__proposals_counter += 1
        proposed_estimator.fit(X, y)
        return proposed_estimator

    def has_next(self) -> bool:
        """
        Method that indicates whether there is a next estimator
        in the sequence.

        Returns:
            bool: True if next estimator can be provided. False otherwise
        """
        return self.__proposals_counter != len(self.__estimators)

    def reset_generator(self) -> None:
        """
        Methods that resets value of proposal counter field to 0.
        """
        self.__proposals_counter = 0
