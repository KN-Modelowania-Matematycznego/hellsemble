from argparse import ArgumentParser, Namespace
from itertools import product

from sklearn.metrics import accuracy_score

from bin.utils.exp_config import MODELS, ROUTERS
from hellsemble.estimator_generator import PredefinedEstimatorsGenerator
from hellsemble.prediction_generator import FixedThresholdPredictionGenerator
from testing.experiment import HellsembleExperiment


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(
    idx,
    models,
    routing_model,
    metric,
    estimators_generator,
    prediction_generator,
    automl,
    experiment_type,
):
    args = get_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    output_dir = args.output_dir
    output_dir += f"_{idx:03}"

    experiment = HellsembleExperiment(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        models=models,
        routing_model=routing_model,
        metric=metric,
        estimators_generator=estimators_generator,
        prediction_generator=prediction_generator,
        automl=automl,
        experiment_type=experiment_type,
    )
    experiment.run()


if __name__ == "__main__":
    estimators_generator = PredefinedEstimatorsGenerator
    prediction_generator = FixedThresholdPredictionGenerator(0.5)

    # Define the metric used to evaluate the models.
    metric = accuracy_score

    automl = None  # set to AutoSklearnRun or AutoGluonRun to use AutoML
    experiment_type = "full"

    for idx, (models, routing_model) in enumerate(product(MODELS, ROUTERS)):
        main(
            idx,
            models,
            routing_model,
            metric,
            estimators_generator,
            prediction_generator,
            automl,
            experiment_type,
        )
