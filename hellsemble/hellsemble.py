from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .estimator_generator import EstimatorGenerator
from .prediction_generator import PredictionGenerator


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
         mode (str): Mode of ensemble creation. Two possible options:
            'greedy' (default) and 'sequential'. 'greedy' mode dynamically
            selects the best estimator based on performance during fitting,
            while 'sequential' mode fits all estimators
            in a predefined sequence.
        metric (callable | Literal["accuracy", "balanced_accuracy",
                                            "roc_auc", "f1"]):
            Metric to evaluate the ensemble's fitting performance.
            Can be one of the predefined string metrics: 'accuracy',
            'balanced_accuracy', 'roc_auc', 'f1', or a custom
            scoring function that accepts `y_true` and `y_pred_proba`
            as arguments.
    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        estimator_generator: EstimatorGenerator,
        prediction_generator: PredictionGenerator,
        routing_model: ClassifierMixin,
        mode: Literal["greedy", "sequential"] = "greedy",
        metric: (
            Callable
            | Literal["accuracy", "balanced_accuracy", "roc_auc", "f1"]
        ) = "accuracy",
    ):
        self.estimator_generator = estimator_generator
        self.routing_model = routing_model
        self.prediction_generator = prediction_generator
        self.mode = mode
        self.metric = metric

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        validation_size: float = 0.25,
        stopping_threshold: float = 0.95,
        seed: int | None = None,
    ) -> Hellsemble:
        """
        Fits the Hellsemble model to the provided data according
        to training mode. When mode is not greedy,
        this method iterates through the estimator generator,
        training each estimator on the data for which the previous
        estimators failed to make correct predictions.
        When mode is greedy, additionally in each iteration the model
        that maximizes F1 score on validation set in current setup
        is selected to ensemble. Method also trains the routing
        model to predict which estimator best suits each data point.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Returns:
            Hellsemble: The fitted Hellsemble model.
        """
        (
            X_train,
            X_val,
            y_train,
            y_val,
            train_idx,
            validation_idx,
        ) = train_test_split(
            X,
            y,
            np.arange(len(X)),
            test_size=validation_size,
            random_state=seed,
        )
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if self.mode == "greedy":
            self.__train_fitting_history, self.__validation_fitting_history, self.coverage_counts, self.performance_scores = (
                self.__fit_estimators_greedy(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    stopping_threshold,
                )
            )
        else:
            (
                self.__train_fitting_history,
                self.__validation_fitting_history,
                self.coverage_counts,
                self.performance_scores,
            ) = self.__fit_estimators_sequential(
                X_train,
                y_train,
                X_val,
                y_val,
                stopping_threshold,
            )

        if len(self.estimators) > 1:

            n_estimators = len(self.__train_fitting_history)
            n_total = X.shape[0]
            full_fitting_history = []
            for i in range(n_estimators):
                full_mask = np.zeros(n_total, dtype=bool)
                full_mask[train_idx] = self.__train_fitting_history[i]
                full_mask[validation_idx] = self.__validation_fitting_history[
                    i
                ]
                full_fitting_history.append(full_mask)

            self.routing_model = self.__fit_routing_model(
                self.routing_model, X, full_fitting_history
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
        if len(self.estimators) == 1:
            return self.estimators[0].predict(X)
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
        if len(self.estimators) == 1:
            return self.estimators[0].predict_proba(X)
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

    def __fit_estimators_sequential(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_validation: np.ndarray | pd.DataFrame,
        y_validation: np.ndarray | pd.Series,
        threshold: float,
    ) -> Tuple[list[ClassifierMixin], list[np.ndarray], list[np.ndarray]]:
        """
        Fits a sequence of estimators and tracks their performance.
        This method iterates through the estimator generator, fitting
        each estimator on the data for which the previous estimators
        failed to make correct predictions.

        Args:
            X_train (np.ndarray | pd.DataFrame): The training data
            y_train (np.ndarray | pd.Series): The training target 
            X_validation (np.ndarray | pd.DataFrame): The data to validate
            y_validation (np.ndarray | pd.Series): The validation target 
            threshold (float): Threshold of main metric that leads to fitting stop

        Returns:
            Tuple[list[ClassifierMixin], list[np.ndarray], list[float], list[float]]: 
                A tuple containing: 
                - list of masks indicating which observations were used during fitting,
                - list of masks indicating which observations were correctly predicted in validation,
                - list of intermediate coverage of training set during iterations of fitting.
                - list of intermediate hellsemble scores during fitting. 
        """
        train_fitting_history: list[np.ndarray] = []
        validation_fitting_history: list[np.ndarray] = []
        self.estimators = []
        self.meta = []
        coverage_counts = []
        performance_scores = []
        failed_observations_idx_fit = np.arange(X_train.shape[0])
        failed_observations_idx_val = np.arange(X_validation.shape[0])

        # Creating copies of data for collecting fitting history
        X_fit, y_fit = X_train, y_train
        X_val, y_val = X_validation, y_validation
        while (
            self.estimator_generator.has_next()
            and not self.__fitting_stop_condition(validation_fitting_history)
            and len(np.unique(y_fit)) > 1
        ):
            # Generate next iterator
            estimator = self.estimator_generator.fit_next_estimator(
                X_fit, y_fit
            )
            self.estimators.append(estimator)

            # Make and evaluate predictions on training set
            fit_predictions = self.prediction_generator.make_prediction_train(
                estimator, X_fit
            )
            failed_observations_mask_fit = (fit_predictions != y_fit)
            coverage_counts.append(X_fit.shape[0] - failed_observations_mask_fit.sum())

            # Make and evaluate predictions on validation set
            val_predictions = self.prediction_generator.make_prediction_train(
                estimator, X_val
            )
            failed_observations_mask_val = (val_predictions != y_val)

            # Create prediction history entry
            failed_observations_idx_fit = failed_observations_idx_fit[
                failed_observations_mask_fit
            ]
            train_fitting_history_entry = np.full((X_train.shape[0]), False)
            train_fitting_history_entry[failed_observations_idx_fit] = True
            train_fitting_history.append(train_fitting_history_entry)

            failed_observations_idx_val = failed_observations_idx_val[
                failed_observations_mask_val
            ]
            validation_fitting_history_entry = np.full(
                (X_validation.shape[0]), False
            )
            validation_fitting_history_entry[failed_observations_idx_val] = (
                True
            )
            validation_fitting_history.append(validation_fitting_history_entry)

            # Update fitting and validation data
            X_fit, y_fit = (
                X_fit[failed_observations_mask_fit],
                y_fit[failed_observations_mask_fit],
            )

            X_val, y_val = (
                X_val[failed_observations_mask_val],
                y_val[failed_observations_mask_val],
            )

            # Validate the ensemble
            if len(self.estimators) > 1:
                self.routing_model = self.__fit_routing_model(
                    self.routing_model,
                    X_train,
                    train_fitting_history + [train_fitting_history_entry],
                )
            val_score = self.evaluate_hellsemble(X_validation, y_validation)
            self.meta.append(val_score)
            performance_score = self.evaluate_hellsemble(X_fit, y_fit)
            performance_scores.append(performance_score)
            if val_score >= threshold:
                break
        return train_fitting_history, validation_fitting_history, coverage_counts, performance_scores

    def __fit_estimators_greedy(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_validation: np.ndarray | pd.DataFrame,
        y_validation: np.ndarray | pd.Series,
        threshold: float,
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Fits a sequence of estimators and tracks their performance.
        It uses the logic of fitting subsequent classifiers on observations
        that previous models failed to predict correctly. But in each iteration
        it dynamically verifies which model improves F! score
        on the validation dataset the most and adds it to estimator list.

        Args:
            X_train (np.ndarray | pd.DataFrame): The training data
            y_train (np.ndarray | pd.Series): The training target 
            X_validation (np.ndarray | pd.DataFrame): The data to validate
            y_validation (np.ndarray | pd.Series): The validation target 
            threshold (float): Threshold of main metric that leads to fitting stop

        Returns:
            Tuple[list[ClassifierMixin], list[np.ndarray], list[float], list[float]]: 
                A tuple containing: 
                - list of masks indicating which observations were used during fitting,
                - list of masks indicating which observations were correctly predicted in validation,
                - list of intermediate coverage of training set during iterations of fitting.
                - list of intermediate hellsemble scores during fitting. 
        """
        train_fitting_history: list[np.ndarray] = []
        validation_fitting_history: list[np.ndarray] = []
        self.estimators = []
        self.meta = []
        coverage_counts = []
        performance_scores = []
        failed_observations_idx_fit = np.arange(X_train.shape[0])
        failed_observations_idx_val = np.arange(X_validation.shape[0])

        # Creating copies of data for collecting fitting history
        X_fit, y_fit = X_train, y_train
        X_val, y_val = X_validation, y_validation
        best_score = 0

        while not self.__fitting_stop_condition(validation_fitting_history):
            best_model = None
            best_ensemble_score = best_score
            self.estimator_generator.reset_generator()

            # Going through provided model list
            while (
                self.estimator_generator.has_next()
                and len(np.unique(y_fit)) > 1
            ):
                estimator = self.estimator_generator.fit_next_estimator(
                    X_fit, y_fit
                )
                predictions = self.prediction_generator.make_prediction_train(
                    estimator, X_fit
                )
                failed_observations_mask_fit = (predictions != y_fit)
                failed_observations_idx_fit_temp = failed_observations_idx_fit[
                    failed_observations_mask_fit
                ]

                train_fitting_history_entry = np.full((X_train.shape[0]), False)
                train_fitting_history_entry[failed_observations_idx_fit_temp] = True
                self.estimators.append(estimator)

                if len(self.estimators) > 1:
                    self.routing_model = self.__fit_routing_model(
                        self.routing_model,
                        X_train,
                        train_fitting_history + [train_fitting_history_entry],
                    )
                # predictions = self.predict(X)
                current_score = self.evaluate_hellsemble(
                    X_validation, y_validation
                )
                self.estimators.pop()

                if current_score >= best_ensemble_score:
                    best_model = estimator
                    best_ensemble_score = current_score
            # Best model from iteration is added to estimators sequence
            if best_model is not None and best_ensemble_score >= best_score:
                self.estimators.append(clone(best_model).fit(X_fit, y_fit))
                best_score = best_ensemble_score
                fit_predictions = (
                    self.prediction_generator.make_prediction_train(
                        best_model, X_fit
                    )
                )   
                performance_scores.append(self.evaluate_hellsemble(X_fit, y_fit))
                failed_observations_mask_fit = (fit_predictions != y_fit)
                coverage_counts.append(len(X_fit) - failed_observations_mask_fit.sum())

                # Make and evaluate predictions on validation set
                val_predictions = (
                    self.prediction_generator.make_prediction_train(
                        best_model, X_val
                    )
                )
                failed_observations_mask_val = (val_predictions != y_val)
                # Create prediction history entry
                failed_observations_idx_fit = failed_observations_idx_fit[
                    failed_observations_mask_fit
                ]
                train_fitting_history_entry = np.full((X_train.shape[0]), False)
                
                train_fitting_history_entry[failed_observations_idx_fit] = True
                train_fitting_history.append(train_fitting_history_entry)

                failed_observations_idx_val = failed_observations_idx_val[
                    failed_observations_mask_val
                ]
                validation_fitting_history_entry = np.full(
                    (X_validation.shape[0]), False
                )
                validation_fitting_history_entry[
                    failed_observations_idx_val
                ] = True
                validation_fitting_history.append(
                    validation_fitting_history_entry
                )

                # Update fitting and validation data
                X_fit, y_fit = (
                    X_fit[failed_observations_mask_fit],
                    y_fit[failed_observations_mask_fit],
                )

                X_val, y_val = (
                    X_val[failed_observations_mask_val],
                    y_val[failed_observations_mask_val],
                )

                if len(failed_observations_idx_fit) == 0:
                    break
                if best_score >= threshold:
                    break
            else:
                break
        return train_fitting_history, validation_fitting_history, coverage_counts, performance_scores

    def __fitting_stop_condition(
        self, validation_fitting_history: list[np.ndarray]
    ) -> bool:
        # Place for additional stop conditions
        return (
            len(validation_fitting_history) > 0
            and (~validation_fitting_history[-1]).mean() >= 0.95
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

    def evaluate_hellsemble(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ):
        """
        Evaluates hellsemble based on primary metric used
        for training in a greedy mode. It predicts labels
        or probabilities on provided dataset to score them against
        known labels and return calculated value.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Returns:
            np.float64: metric score
        """
        metrics_map = {
            "roc_auc": roc_auc_score,
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "f1": f1_score,
        }

        if callable(self.metric):
            y_pred_proba = self.predict_proba(X)[:, 1]
            return self.metric(y, y_pred_proba)

        if self.metric == "roc_auc":
            y_pred_proba = self.predict_proba(X)[:, 1]
            return metrics_map["roc_auc"](y, y_pred_proba)

        y_pred = self.predict(X)
        return metrics_map[self.metric](y, y_pred)

    def get_progressive_scores(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ):
        """
        Evaluates hellsemble based on primary metric used
        for training in a greedy mode. Calculates the progressive scores of Hellsemble with each added model.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Returns:
            list[float]: List of metric scores for each step in the ensemble.
        """
        try:
            estimators_copy = self.estimators.copy.deepcopy()
            scores = []
            for i in range(1, len(estimators_copy) + 1):
                self.estimators = estimators_copy[:i]
                scores.append(self.evaluate_hellsemble(X, y))
        except Exception as e:
            print(f"An error occurred while calculating progressive scores: {e}")
            scores = []

        return scores

    def evaluate_routing_model(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> float:
        """
        Evaluates the performance of the routing model by comparing
        the routing model's assignments with the actual performance
        of the estimators on the routed data points.

        Args:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (np.ndarray | pd.Series): Target vector

        Returns:
            float: Accuracy of the routing model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if len(self.estimators) == 1:
            return 1.0

        routing_predictions = self.routing_model.predict(X)

        correct_routing = []

        for idx in range(X.shape[0]):
            best_estimator_index = None
            min_error = np.inf

            for i, estimator in enumerate(self.estimators):
                y_pred = estimator.predict(X[idx].reshape(1, -1))
                error = np.abs(y[idx] - y_pred).item()

                if error < min_error:
                    min_error = error
                    best_estimator_index = i
            correct_routing.append(routing_predictions[idx] == best_estimator_index)

        routing_accuracy = np.mean(correct_routing)
        return routing_accuracy