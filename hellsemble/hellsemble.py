from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .estimator_generator import EstimatorGenerator
from .predction_generator import PredictionGenerator


class Hellsemble(BaseEstimator):
    """
    Ensemble classifier that implements Hellsemeble ensembling strategy.
    This class implements a stacked ensemble approach where a sequence of
    estimators are trained one after the other on the data. A routing model
    is used to assign each data point to the most appropriate estimator
    in the sequence. The final prediction is made by the assigned estimator.

    Attributes:
        estimator_generator (EstimatorGenerator): An object that generates
            consecutive estimators used during fit.
        routing_model (ClassifierMixin): The classifier used to
            route data points to specific estimators during prediction.
        prediction_generator (PredictionGenerator): An object that generates
            predictions from the estimators.
    """

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
        """
        Fits the Hellsemble model to the provided data.
        This method iterates through the estimator generator,
        training each estimator on the data for which the previous estimators
        failed to make correct predictions. It also trains the routing model
        to predict which estimator best suits each data point.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Returns:
            Hellsemble: The fitted Hellsemble model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.estimators, self.__fitting_history = self.__fit_estimators(X, y)
        self.routing_model = self.__fit_routing_model(
            self.routing_model, X, self.__fitting_history
        )
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts class labels for unseen data.
        This method utilizes the routing model to assign each data point in X
        to an appropriate estimator in the ensemble. It then uses
        the prediction_generator to obtain predictions
        from the assigned estimators.

        Args:
            X (np.ndarray | pd.DataFrame): The data to predict on.

        Returns:
            np.ndarray: The predicted class labels for the data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        prediction = np.zeros(shape=(X.shape[0]))
        observations_to_classifiers_mapping = self.routing_model.predict(X)
        for i, estimator in enumerate(self.estimators):
            if np.any(observations_to_classifiers_mapping == i):
                prediction[observations_to_classifiers_mapping == i] = (
                    self.prediction_generator.make_prediction(
                        estimator, X[observations_to_classifiers_mapping == i]
                    )
                )

        return prediction

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts class probabilities for unseen data. This method utilizes
        the routing model to assign each data point in X to an appropriate
        estimator in the ensemble. It then uses the predict_proba method
        of the assigned estimators to obtain probability estimates.

        Args:
            X (np.ndarray | pd.DataFrame): The data to predict on

        Returns:
            np.ndarray: The predicted class probabilities for the data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        prediction = np.zeros(shape=(X.shape[0], 2))
        observations_to_classifiers_mapping = self.routing_model.predict(X)
        for i, estimator in enumerate(self.estimators):
            if np.any(observations_to_classifiers_mapping == i):
                prediction[observations_to_classifiers_mapping == i] = (
                    estimator.predict_proba(
                        X[observations_to_classifiers_mapping == i]
                    )
                )
        return prediction

    def __fit_estimators(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> Tuple[list[ClassifierMixin], list[np.ndarray]]:
        """
        Fits a sequence of estimators and tracks their performance.
        This method iterates through the estimator generator, fitting
        each estimator on the data for which the previous estimators
        failed to make correct predictions.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Returns:
            Tuple[list[ClassifierMixin], list[np.ndarray]]: A tuple containing
                list of fitted estimators and the list of masks indicating
                which observations were used during fit of the estimators.
        """
        fitting_history: list[np.ndarray] = []
        output_estimators = []
        failed_observations_idx = np.arange(X.shape[0])
        X_fit, y_fit = X, y
        while (
            self.estimator_generator.has_next()
            and not self.__fitting_stop_condition(fitting_history)
        ):
            # Generate next iterator
            estimator = self.estimator_generator.fit_next_estimator(
                X_fit, y_fit
            )
            output_estimators.append(estimator)

            # Make and evaluate predictions
            estimator_predictions = (
                self.prediction_generator.make_prediction_train(
                    estimator, X_fit
                )
            )
            failed_observations_mask = estimator_predictions != y_fit

            # Create prediction history entry
            failed_observations_idx = failed_observations_idx[
                failed_observations_mask
            ]
            fitting_history_entry = np.full((X.shape[0]), False)
            fitting_history_entry[failed_observations_idx] = True
            fitting_history.append(fitting_history_entry)

            # Update fitting data
            X_fit, y_fit = (
                X_fit[failed_observations_mask],
                y_fit[failed_observations_mask],
            )
        return output_estimators, fitting_history

    def __fitting_stop_condition(
        self, fitting_history: list[np.ndarray]
    ) -> bool:
        # Place for additional stop conditions
        return (
            len(fitting_history) > 0 and (~fitting_history[-1]).mean() >= 0.95
        )

    def __fit_routing_model(
        self,
        routing_model: ClassifierMixin,
        X: np.ndarray | pd.DataFrame,
        fitting_history: list[np.ndarray],
    ) -> ClassifierMixin:
        y = self.__generate_fitting_data_for_routing_model(fitting_history)
        return routing_model.fit(X, y)

    def __generate_fitting_data_for_routing_model(
        self, fitting_history: list[np.ndarray]
    ) -> np.ndarray:
        """
        Generates labels for fitting the routing model based on fitting
        history. This method iterates through the fitting history
        in reverse order and assigns a label to each data point indicating
        which estimator in the sequence was the first to make a correct
        prediction. Data points that were never correctly classified by any
        estimator are assigned a label corresponding to the last estimator.

        Args:
            fitting_history (list[np.ndarray]): Fitting history.

        Returns:
            np.ndarray: The labels for fitting the routing model.
        """
        routing_model_target = -1 * np.ones(shape=(len(fitting_history[0])))
        for i, prediction_history_entry in zip(
            reversed(range(len(fitting_history))),
            reversed(fitting_history),
        ):
            routing_model_target[~prediction_history_entry] = i
        # Remaining observations go to the last model.
        # Maybe they should go somewhere else?
        routing_model_target[routing_model_target == -1] = (
            len(fitting_history) - 1
        )
        return routing_model_target
