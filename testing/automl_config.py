from typing import Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod


class AutoMLConfig(ABC):
    @abstractmethod
    def get_models_from_automl(self) -> list[ClassifierMixin]:
        """
        M<ethod used for getting models from the automl framework.
        """

    @abstractmethod
    def _automl_model_map(self) -> Dict[str, str]:
        """
        Method used for providing map of models returned to the automl framework ro usable sklearn objects.
        """

    @abstractmethod
    def _run_automl(self) -> list[str]:
        """
        Method used for running the automl framework.
        """
