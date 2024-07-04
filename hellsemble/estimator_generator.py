from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


class EstimatorGenerator(ABC):

    @abstractmethod
    def fit_next_estimator(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> ClassifierMixin:
        pass

    @abstractmethod
    def has_next(self) -> bool:
        pass


class PredefinedEstimatorsGenerator(EstimatorGenerator):

    def __init__(self, estimators: list[ClassifierMixin]):
        self.__estimators = estimators
        self.__proposals_counter = 0

    def fit_next_estimator(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> ClassifierMixin:
        if self.__proposals_counter == len(self.__estimators):
            raise StopIteration()
        proposed_estimator = self.__estimators[self.__proposals_counter]
        self.__proposals_counter += 1
        proposed_estimator.fit(X, y)
        return proposed_estimator

    def has_next(self) -> bool:
        return self.__proposals_counter != len(self.__estimators)
