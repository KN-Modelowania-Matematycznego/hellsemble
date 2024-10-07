from experiment import HellsembleExperiment
from hellsemble.estimator_generator import PredefinedEstimatorsGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# from testing.autosklearn_config import AutoSklearnConfig
from testing.autogluon_config import AutoGluonConfig


###
### This script is used to run the experiment for the Hellsemble project.
###

# Define the directories containing the training and test data.
train_dir = "resources/data/openml/train/automl"
test_dir = "resources/data/openml/test/automl"

# Define the directory to save the results to.
output_dir = "resources/data/openml/results/automl"

# Define the base models to train and test.
models = [
    ExtraTreesClassifier(),
    RandomForestClassifier(),
]

# Define the routing model used in the Hellsemble ensemble.
routing_model = KNeighborsClassifier()
estimators_generator = PredefinedEstimatorsGenerator

# Define the metric used to evaluate the models.
metric = accuracy_score
automl = AutoGluonConfig()

experiment = HellsembleExperiment(
    train_dir=train_dir,
    test_dir=test_dir,
    output_dir=output_dir,
    models=models,
    routing_model=routing_model,
    metric=metric,
    estimators_generator=estimators_generator,
    automl=automl,
    experiment_type="full",
)

if __name__ == "__main__":
    experiment.run()
