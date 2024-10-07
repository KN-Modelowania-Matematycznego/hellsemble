from typing import Dict, List, Callable
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from datetime import datetime
import json
import os
from hellsemble.hellsemble import Hellsemble
from hellsemble.estimator_generator import EstimatorGenerator
from hellsemble.predction_generator import FixedThresholdPredictionGenerator
from loguru import logger
from testing.automl_config import AutoMLConfig


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
        metric: Callable,
        estimators_generator: EstimatorGenerator,
        automl: AutoMLConfig = None,
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

    def _get_data_from_file(self, train_file: str, test_file: str):
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        return X_train, X_test, y_train, y_test

    def _train_and_test_base_models(
        self, train_file: str, test_file: str
    ) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = self._get_data_from_file(
            train_file, test_file
        )

        results = {}

        for model in self.models:

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            eval = self.metric(y_test, y_pred)
            results[model.__repr__()] = eval

        return results

    def _train_and_test_hellsemble(
        self, train_file: str, test_file: str, mode: str
    ) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = self._get_data_from_file(
            train_file, test_file
        )

        estimator = Hellsemble(
            self.estimators_generator(self.models),
            FixedThresholdPredictionGenerator(0.5),
            self.routing_model,
            mode=mode,
        )

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        eval = self.metric(y_test, y_pred)

        return {f"Hellsemble_{mode}": eval}

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

        logger.info("Running experiment for:")
        if not self.automl:
            logger.info(f"  Models: {self.models}")
        else:
            logger.info("  Using AutoML to generate models for each data set.")

        results = {}

        train_files = [
            os.path.join(self.train_dir, f)
            for f in os.listdir(self.train_dir)
            if f.endswith(".csv")
        ]
        test_files = [
            os.path.join(self.test_dir, f)
            for f in os.listdir(self.test_dir)
            if f.endswith(".csv")
        ]

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
                    results[dataset_name]["base_models"] = (
                        self._train_and_test_base_models(train_file, test_file)
                    )
                except Exception as e:
                    logger.error(
                        f"Error running base models experiment for dataset {dataset_name}: {e}"
                    )
            if self.experiment_type in ["full", "hellsemble"]:
                # TODO: Change 'sequential' and 'greedy' modes to not be hardcoded strings, but to iterate over all possible modes.

                results[dataset_name]["hellsemble"] = {}

                try:
                    results[dataset_name]["hellsemble"].update(
                        self._train_and_test_hellsemble(
                            train_file, test_file, "Sequential"
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error running Sequential Hellsemble experiment for dataset {dataset_name}: {e}"
                    )
                try:
                    results[dataset_name]["hellsemble"].update(
                        self._train_and_test_hellsemble(train_file, test_file, "greedy")
                    )
                except Exception as e:
                    logger.error(
                        f"Error running greedy Hellsemble experiment for dataset {dataset_name}: {e}"
                    )

        logger.info(f"Saving results to {self.output_dir}")

        with open(f"{self.output_dir}/experiment_results.json", "w") as json_file:
            json.dump(results, json_file)

        with open(f"{self.output_dir}/experiment_config.json", "w") as json_file:
            json.dump(self._create_experiment_config(), json_file)

        logger.info("Experiment complete.")
