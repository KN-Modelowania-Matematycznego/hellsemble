from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .estimator_generator import EstimatorGenerator
from .predction_generator import PredictionGenerator


class Hellsemble(BaseEstimator):

    def __init__(
        self,
        estimator_generator: EstimatorGenerator,
        prediction_generator: PredictionGenerator,
        routing_model: ClassifierMixin,
    ):
        self.estimator_generator = estimator_generator
        self.routing_model = routing_model
        self.prediction_generator = prediction_generator

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> Hellsemble:
        self.estimators, self.__correct_predictions_history = (
            self.__fit_estimators(X, y)
        )
        self.routing_model = self.__fit_routing_model(
            self.routing_model, X, self.__correct_predictions_history
        )
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        prediction = np.zeros(shape=(X.shape[0], 2))
        observations_to_classifiers_mapping = self.routing_model.predict(X)
        for i, estimator in enumerate(self.estimators):
            prediction[observations_to_classifiers_mapping == i] = (
                self.prediction_generator.make_prediction(
                    estimator, X.loc[observations_to_classifiers_mapping == i]
                )
            )

        return prediction

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        prediction = np.zeros(shape=(X.shape[0], 2))
        observations_to_classifiers_mapping = self.routing_model.predict(X)
        for i, estimator in enumerate(self.estimators):
            prediction[observations_to_classifiers_mapping == i] = (
                estimator.predict_proba(
                    X.loc[observations_to_classifiers_mapping == i]
                )
            )
        return prediction

    def __fit_estimators(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> Tuple[list[ClassifierMixin], list[np.ndarray]]:
        correct_predictions_history: list[np.ndarray] = []
        output_estimators = []
        misclassified_observations_idx = np.arange(X.shape[0])
        X_fit, y_fit = X, y
        while self.estimator_generator.has_next():
            # Generate next iterator
            estimator = self.estimator_generator.fit_next_estimator(
                X_fit, y_fit
            )
            output_estimators.append(estimator)

            # Make and evaluate predictions
            estimator_predictions = self.prediction_generator.make_prediction(
                estimator, X_fit
            )
            correct_predictions_mask = estimator_predictions == y_fit

            # Create prediction history entry
            misclassified_observations_idx = misclassified_observations_idx[
                ~correct_predictions_mask
            ]
            prediction_history_entry = np.full((X.shape[0]), True)
            prediction_history_entry[misclassified_observations_idx] = False
            correct_predictions_history.append(prediction_history_entry)

            # Update fitting data
            X_fit, y_fit = (
                X_fit[~correct_predictions_mask],
                y_fit[~correct_predictions_mask],
            )
        return output_estimators, correct_predictions_history

    def __fit_routing_model(
        self,
        routing_model: ClassifierMixin,
        X: np.ndarray | pd.DataFrame,
        correct_predictions_history: list[np.ndarray],
    ) -> ClassifierMixin:
        y = self.__generate_fitting_data_for_routing_model(
            correct_predictions_history
        )
        return routing_model.fit(X, y)

    def __generate_fitting_data_for_routing_model(
        self, correct_predictions_history: list[np.ndarray]
    ) -> np.ndarray:
        correct_prediction_model_idx = -1 * np.ones(
            shape=(len(correct_predictions_history[0]))
        )
        for i, prediction_history_entry in zip(
            reversed(range(len(correct_predictions_history))),
            reversed(correct_predictions_history),
        ):
            correct_prediction_model_idx[prediction_history_entry] = i
        # Remaining observations go to the last canto.
        # Maybe they should go somewhere else?
        correct_prediction_model_idx[correct_prediction_model_idx == -1] = (
            len(correct_predictions_history) - 1
        )
        return correct_prediction_model_idx
