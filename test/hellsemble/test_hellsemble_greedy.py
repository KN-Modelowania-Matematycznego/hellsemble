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
from hellsemble.predction_generator import (
    FixedThresholdPredictionGenerator,
    PredictionGenerator,
)


@pytest.fixture
def estimator_generator() -> EstimatorGenerator:
    estimators = [Mock(spec=ClassifierMixin) for _ in range(3)]
    for est in estimators:
        est.fit = Mock(return_value=est)
        est.predict = Mock(return_value=np.random.randint(0, 2, 100))

    generator = PredefinedEstimatorsGenerator(estimators)
    generator.reset_generator = Mock()
    generator.has_next = Mock(side_effect=[True, True, False])
    generator.fit_next_estimator = Mock(side_effect=estimators)
    return generator


@pytest.fixture
def prediction_generator() -> PredictionGenerator:
    pg = FixedThresholdPredictionGenerator(0.5)
    pg.make_prediction_train = Mock()
    return pg


@pytest.fixture
def routing_model() -> ClassifierMixin:
    model = Mock(spec=ClassifierMixin)
    model.fit = Mock(return_value=model)
    return model


@pytest.fixture
def train_data() -> Tuple[np.ndarray, np.ndarray]:
    return np.random.randn(100, 10), np.random.randint(0, 2, 100)


def test__fit_estimators_greedy(
    train_data, estimator_generator, prediction_generator, routing_model
):
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )
    X, y = train_data
    initial_estimators_count = 0

    prediction_generator.make_prediction_train.return_value = y
    hellsemble._Hellsemble__fit_estimators_greedy(X, y)

    assert len(hellsemble.estimators) == initial_estimators_count + 1
    assert (
        f1_score(
            y,
            hellsemble.prediction_generator.make_prediction_train(
                hellsemble.estimators[-1], X
            ),
        )
        == 1.0
    )
