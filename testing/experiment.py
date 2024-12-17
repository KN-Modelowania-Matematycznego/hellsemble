from typing import Dict, List, Callable, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from testing.test_hellsemble.hellsemble import Hellsemble
from testing.test_hellsemble.estimator_generator import EstimatorGenerator
from testing.test_hellsemble.prediction_generator import (
    FixedThresholdPredictionGenerator,
)
from loguru import logger
from testing.automl_config import AutoMLRun
from testing.eval_utils import calculate_ranks, calculate_adtm, generate_CD_plot
from pathlib import Path


class HellsembleExperiment:
    """
    A class for runing Hellsemble experiments.
    Takes in a set of base models which are fitted to the training data individually,
    as well as in a Hellsemble ensemble, for all specified model selection methods.
    Experiments are ran for all binary classification datasets in the specified directories.

    Args:
    train_dir: str
        The directory containing the training data.
    test_dir: str
        The directory containing the test data.
    output_dir: str
        The directory to save the results to.
    models: List[ClassifierMixin]
        A list of base models to train and test.
    routing_model: ClassifierMixin
        The model used to route the base models in the Hellsemble ensemble.
    metric: Callable
        The metric used to evaluate the models.
    estimators_generator: EstimatorGenerator
        The generator used to create the base models for the Hellsemble ensemble.
    automl: bool
        Whether to use autmoml to generate the model set instead of passing in a list of models.
    experiment_type: str
        The type of experiment to run. Options are 'full', 'base_models', 'hellsemble'.
    """

    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        output_dir: str,
        models: List[ClassifierMixin],
        routing_model: ClassifierMixin,
        metric: Callable[[np.ndarray, np.ndarray], float],
        estimators_generator: EstimatorGenerator,
        prediction_generator: FixedThresholdPredictionGenerator,
        automl: AutoMLRun = None,
        experiment_type: str = "full",
    ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.models = models
        self.routing_model = routing_model
        self.metric = metric
        self.estimators_generator = estimators_generator
        self.automl = automl
        self.experiment_type = experiment_type
        self.prediction_generator = prediction_generator

    def _get_data_from_file(self, train_file: str, test_file: str):
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        return X_train, X_test, y_train, y_test

    def _get_base_model_results(
        self, train_file: str, test_file: str
    ) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = self._get_data_from_file(
            train_file, test_file
        )

        results = {}

        for model in self.models:

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluation = self.metric(y_test, y_pred)
            results[model.__repr__()] = evaluation

        return results

    def _train_and_test_hellsemble(
        self, train_file: str, test_file: str, mode: str
    ) -> Tuple[Dict[str, float]]:
        X_train, X_test, y_train, y_test = self._get_data_from_file(
            train_file, test_file
        )

        estimator = Hellsemble(
            self.estimators_generator(self.models),
            self.prediction_generator,
            self.routing_model,
            mode=mode,
        )

        estimator.fit(X_train, y_train)
        eval = estimator.evaluate_hellsemble(X_test, y_test)
        hellsemble_estimators = estimator.estimators
        routing_accuracy = estimator.evaluate_routing_model(X_test, y_test)
        eval_scores = estimator.get_progressive_scores(X_test, y_test)

        logger.info(f"Models selected by {mode} Hellsemble: {hellsemble_estimators}")
        return {f"Hellsemble_{mode}": eval}, {
            "score": eval,
            "num_models": len(hellsemble_estimators),
            "progressive_scores": eval_scores,
            "routing_accuracy": routing_accuracy,
            "models": {
                str(model): {
                    "coverage_perc": estimator.coverage_counts[i] / len(X_train),
                    "performance_score": estimator.performance_scores[i],
                }
                for i, model in enumerate(hellsemble_estimators)
            },
        }

    def _create_experiment_config(self):
        return {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "automl": self.automl.__class__.__name__,
            "models": [str(model) for model in self.models],
            "routing_model": str(self.routing_model),
            "estimators_generator": str(self.estimators_generator),
            "metric": self.metric.__name__,
            "experiment_type": self.experiment_type,
        }

    def run(self):

        if not self.automl:
            logger.info(f"Running experiment for models: {self.models}")
        else:
            logger.info(
                " Running experiment and using AutoML to generate models for each data set."
            )

        results = {}
        hellsemble_results_info = {}

        train_files = list(Path(self.train_dir).rglob("*.csv"))
        test_files = list(Path(self.test_dir).rglob("*.csv"))

        for train_file, test_file in zip(train_files, test_files):
            dataset_name = os.path.basename(train_file).replace(".csv", "")
            if self.automl:
                train_data = pd.read_csv(train_file)
                self.models = self.automl.get_models_from_automl(train_data)
                logger.info(
                    f"Selected models for dataset {dataset_name}: {self.models}"
                )
            results[dataset_name] = {}

            logger.info(f"Running experiment for dataset: {dataset_name}")
            if self.experiment_type in ["full", "base_models"]:
                try:
                    results[dataset_name]["base_models"] = self._get_base_model_results(
                        train_file, test_file
                    )
                except Exception as e:
                    logger.error(
                        f"Error running base models experiment for dataset {dataset_name}: {e}"
                    )
            if self.experiment_type in ["full", "hellsemble"]:
                results[dataset_name]["hellsemble"] = {}
                hellsemble_results_info[dataset_name] = {"greedy": {}, "sequential": {}}
                try:
                    run_results = self._train_and_test_hellsemble(
                        train_file, test_file, "sequential"
                    )
                    results[dataset_name]["hellsemble"].update(run_results[0])
                    hellsemble_results_info[dataset_name]["sequential"].update(
                        run_results[1]
                    )
                except Exception as e:
                    logger.error(
                        f"Error running sequential Hellsemble experiment for dataset {dataset_name}: {e}"
                    )
                try:
                    run_results = self._train_and_test_hellsemble(
                        train_file, test_file, "greedy"
                    )
                    results[dataset_name]["hellsemble"].update(run_results[0])
                    hellsemble_results_info[dataset_name]["greedy"].update(
                        run_results[1]
                    )
                except Exception as e:
                    logger.error(
                        f"Error running greedy Hellsemble experiment for dataset {dataset_name}: {e}"
                    )

        average_ranks, ranks_df = calculate_ranks(results)

        logger.info(f"Average ranks = {average_ranks}")
        generate_CD_plot(average_ranks, ranks_df)

        logger.info(f"Saving results to {self.output_dir}")
        result_eval = calculate_adtm(results)
        logger.info(f"ADTM score = {result_eval}")

        results["average_ranks"] = average_ranks
        results["ADTM"] = result_eval

        with open(f"{self.output_dir}/experiment_results.json", "w") as json_file:
            json.dump(results, json_file)
        with open(
            f"{self.output_dir}/experiment_hellsemble_info.json", "w"
        ) as json_file:
            json.dump(hellsemble_results_info, json_file)

        with open(f"{self.output_dir}/experiment_config.json", "w") as json_file:
            json.dump(self._create_experiment_config(), json_file)

        logger.info("Experiment complete.")
