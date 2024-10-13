from testing.experiment import HellsembleExperiment
from testing.test_hellsemble.estimator_generator import PredefinedEstimatorsGenerator
from testing.test_hellsemble.prediction_generator import (
    FixedThresholdPredictionGenerator,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import accuracy_score
from testing.autogluon_config import AutoGluonRun
from testing.autosklearn_config import AutoSklearnRun


def main(
    train_dir,
    test_dir,
    output_dir,
    models,
    routing_model,
    metric,
    estimators_generator,
    prediction_generator,
    automl,
    experiment_type,
):
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
    # Define the directories containing the training and test data.
    train_dir = "resources/data/openml/train"
    test_dir = "resources/data/openml/test"

    # Define the directory to save the results to.
    output_dir = "resources/data/openml/results"

    # Define the base models to train and test.
    models = [
        ExtraTreesClassifier(),
        RandomForestClassifier(),
    ]

    # Define the routing model used in the Hellsemble ensemble.
    routing_model = KNeighborsClassifier()
    estimators_generator = PredefinedEstimatorsGenerator
    prediction_generator = FixedThresholdPredictionGenerator(0.5)

    # Define the metric used to evaluate the models.
    metric = accuracy_score

    automl = None  # set to AutoSklearnRun or AutoGluonRun to use AutoML
    experiment_type = "full"

    main(
        train_dir,
        test_dir,
        output_dir,
        models,
        routing_model,
        metric,
        estimators_generator,
        prediction_generator,
        automl,
        experiment_type,
    )
