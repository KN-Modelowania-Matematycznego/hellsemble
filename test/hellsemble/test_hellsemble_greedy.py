from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score

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
    estimators = [Mock(spec=ClassifierMixin) for _ in range(3)]
    for est in estimators:
        est.fit = Mock(return_value=est)
    estimators[0].predict = Mock(
        return_value=np.concatenate((np.zeros(10), np.ones(90)))
    )
    estimators[1].predict = Mock(
        return_value=np.concatenate((np.zeros(50), np.ones(49), np.zeros(1)))
    )
    estimators[2].predict = Mock(
        return_value=np.concatenate((np.zeros(20), np.ones(80)))
    )

    generator = PredefinedEstimatorsGenerator(estimators)
    generator.reset_generator = Mock()
    generator.has_next = Mock(side_effect=[True, True, True, False])
    # generator.has_next = Mock(return_value=True)
    generator.fit_next_estimator = Mock(side_effect=estimators)
    return generator


@pytest.fixture
def prediction_generator() -> PredictionGenerator:
    mock_predictions = [
        np.concatenate((np.zeros(10), np.ones(90))),
        np.concatenate((np.zeros(50), np.ones(49), np.zeros(1))),
        np.concatenate((np.zeros(20), np.ones(80))),
        np.concatenate((np.zeros(50), np.ones(49), np.zeros(1))),
    ]
    mock_predictions_generator = (pred for pred in mock_predictions)

    def prediction_generator_side_effect(*args, **kwargs) -> np.ndarray:
        return next(mock_predictions_generator)

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
    return np.random.randn(100, 10), np.concatenate(
        (np.zeros(50), np.ones(50))
    )


def test__fit_estimators_greedy(
    train_data, estimator_generator, prediction_generator, routing_model
):
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )
    X, y = train_data
    hellsemble._Hellsemble__fit_estimators_greedy(X, y)
    predictions = hellsemble.predict(X)
    assert len(hellsemble.estimators) == 1
    assert (predictions == y).sum() == 99
    assert round(f1_score(y, predictions), 2) == 0.99
