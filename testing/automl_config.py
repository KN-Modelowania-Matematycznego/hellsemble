from typing import Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod


class AutoMLRun(ABC):
    @property
    def get_models_from_automl(self) -> list[ClassifierMixin]:
        """
        Method used for getting models from the automl framework.
        """

    @abstractmethod
    def automl_model_map(self) -> Dict[str, str]:
        """
        Method used for providing map of models returned to the automl framework or usable sklearn objects.
        """

    @abstractmethod
    def run_automl(self) -> list[str]:
        """
        Method used for running the automl framework.
        """
