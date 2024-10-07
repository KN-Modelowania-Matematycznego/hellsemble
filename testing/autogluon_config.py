from typing import Dict
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from testing.automl_config import AutoMLConfig
from autogluon.tabular import TabularDataset, TabularPredictor


class AutoGluonConfig(AutoMLConfig):
    """
    A class for creating a automl configuration for our hellse,ble experiment.
    Right now set to fit autogluon ensemble.
    """

    def __init__(
        self,
        predefined_model_set: Dict[str, str] = {
            "GBM": {},  # LightGBM
            "CAT": {},  # CatBoost
            "XGB": {},  # XGBoost
            "XT": {},  # Extra Trees
            "LR": {},  # Logistic Regression
            "RF": {},  # Random Forest
            "KNN": {},  # k-Nearest Neighbors
        },
        num_stack_levels: int = 1,
        num_bag_folds: int = 5,
    ):
        self.predefined_model_set = predefined_model_set
        self.num_stack_levels = num_stack_levels
        self.num_bag_folds = num_bag_folds

    def _automl_model_map(self) -> Dict[str, str]:
        full_model_set = {
            "LightGBM": LGBMClassifier(),  # LightGBM
            "CatBoost": CatBoostClassifier(),  # CatBoost
            "XGBoost": XGBClassifier(),  # XGBoost
            "ExtraTrees": ExtraTreesClassifier(),  # Extra Trees
            "LinearModel": LogisticRegression(),  # Logistic Regression
            "RandomForest": RandomForestClassifier(),  # Random Forest
            "KNeighbors": KNeighborsClassifier(),  # k-Nearest Neighbors
        }
        return full_model_set

    def _run_automl(self, train_data: pd.DataFrame) -> list[str]:

        predictor = TabularPredictor(
            label="target",
            path="testing/__pycache__",
        ).fit(
            train_data=train_data,
            hyperparameters=self.predefined_model_set,
            time_limit=60,
            num_stack_levels=self.num_stack_levels,
            num_bag_folds=self.num_bag_folds,
            verbosity=0,
        )

        info = predictor.info()["model_info"]
        best_ensemble = info[predictor.get_model_best()]
        models = best_ensemble["children_info"]["S1F1"]["model_weights"].keys()
        models = list(models)
        models = [model.split("_")[0] for model in models]
        models = list(set(models))

        return models

    def get_models_from_automl(self, train_data: pd.DataFrame) -> list[ClassifierMixin]:
        model_map = self._automl_model_map()
        models = self._run_automl(train_data=train_data)
        model_objects = [model_map[model] for model in models if model in model_map]
        return model_objects
