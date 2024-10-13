from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

try:
    import autosklearn.classification
except ModuleNotFoundError:
    Warning(
        "AutoSklearn is not installed. Some functionalities may not work.",
        ImportWarning,
    )
from testing.automl_config import AutoMLRun
from typing import Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from loguru import logger


class AutoSklearnRun(AutoMLRun):
    def __init__(
        self,
        time_left_for_this_task: int = 120,
        per_run_time_limit: int = 30,
        ensemble_size: int = 50,
    ):
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.ensemble_size = ensemble_size

    def automl_model_map(self) -> Dict[str, str]:
        return {
            "adaboost": AdaBoostClassifier(),
            "bernoulli_nb": BernoulliNB(),
            "decision_tree": DecisionTreeClassifier(),
            "extra_trees": ExtraTreesClassifier(),
            "gaussian_nb": GaussianNB(),
            "gradient_boosting": GradientBoostingClassifier(),
            "k_nearest_neighbors": KNeighborsClassifier(),
            "lda": LinearDiscriminantAnalysis(),
            "liblinear_svc": LinearSVC(),
            "libsvm_svc": SVC(),
            "mlp": MLPClassifier(),
            "multinomial_nb": MultinomialNB(),
            "passive_aggressive": PassiveAggressiveClassifier(),
            "qda": QuadraticDiscriminantAnalysis(),
            "random_forest": RandomForestClassifier(),
            "sgd": SGDClassifier(),
        }

    def run_automl(self, train_data: pd.DataFrame) -> list[str]:
        logger.info(
            f"Time set for single AutoSklearn run: {self.time_left_for_this_task}"
        )
        X = train_data.drop(columns=["target"])
        y = train_data["target"]

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            n_jobs=-1,
            ensemble_size=5,
            memory_limit=None,
        )
        automl.fit(X, y)

        models = automl.leaderboard(detailed=True).type.to_list()

        return models

    def get_models_from_automl(self, train_data: pd.DataFrame) -> list[ClassifierMixin]:
        model_map = self.automl_model_map()
        models = self.run_automl(train_data=train_data)
        model_objects = [model_map[model] for model in models if model in model_map]
        return model_objects
