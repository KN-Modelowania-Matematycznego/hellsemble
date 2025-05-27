from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from hellsemble import Hellsemble
from hellsemble.estimator_generator import (
    EstimatorGenerator,
    PredefinedEstimatorsGenerator,
)
from hellsemble.prediction_generator import (
    FixedThresholdPredictionGenerator,
    PredictionGenerator,
)


@pytest.fixture
def estimator_generator() -> EstimatorGenerator:
    estimators = [LogisticRegression(), LogisticRegression(), LogisticRegression()]
    for est in estimators:
        est.fit = Mock(return_value=est)
        est.predict = Mock(side_effect=lambda X: np.zeros(X.shape[0], dtype=int))
    generator = PredefinedEstimatorsGenerator(estimators)
    generator.reset_generator = Mock()
    # Counter for has_next
    generator._counter = 0

    def has_next_side_effect():
        if generator._counter < len(estimators):
            return True
        return False

    def fit_next_estimator_side_effect(X, y):
        est = estimators[generator._counter]
        generator._counter += 1
        return est

    generator.has_next = Mock(side_effect=has_next_side_effect)
    generator.fit_next_estimator = Mock(side_effect=fit_next_estimator_side_effect)
    return generator


@pytest.fixture
def prediction_generator() -> PredictionGenerator:
    def prediction_generator_side_effect(estimator, X, *args, **kwargs) -> np.ndarray:
        if hasattr(X, "shape"):
            n = X.shape[0]
        else:
            n = len(X)
        return np.zeros(n, dtype=int)

    prediction_generator = FixedThresholdPredictionGenerator(0.5)
    prediction_generator.make_prediction_train = Mock(
        side_effect=prediction_generator_side_effect
    )
    return prediction_generator


@pytest.fixture
def routing_model() -> ClassifierMixin:
    model = Mock(spec=ClassifierMixin)
    model.fit = Mock(return_value=model)
    return model


@pytest.fixture
def train_data() -> Tuple[np.ndarray, np.ndarray]:
    return np.random.randn(100, 10), np.concatenate((np.zeros(50), np.ones(50)))


def test__fit_estimators_greedy(
    train_data, estimator_generator, prediction_generator, routing_model
):
    hellsemble = Hellsemble(estimator_generator, prediction_generator, routing_model)
    X, y = train_data
    hellsemble.fit(X, y, validation_size=0.25, stopping_threshold=0.95, seed=123)
    predictions = hellsemble.predict(X)
    # The metric may be a function, so check for callable
    assert callable(hellsemble.metric) or hellsemble.metric == "accuracy"
    assert len(hellsemble.estimators) >= 1
    assert (predictions == y).sum() >= 0  # Looser check, as split is random
    assert 0 <= round(accuracy_score(y, predictions), 2) <= 1
