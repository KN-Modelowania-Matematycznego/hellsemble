from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from hellsemble import Hellsemble
from hellsemble.estimator_generator import (
    EstimatorGenerator,
    PredefinedEstimatorsGenerator,
)
from hellsemble.predction_generator import (
    FixedThresholdPredictionGenerator,
    PredictionGenerator,
)


def assert_called_once_with_numpy_arrays(
    mock: Mock, expected_arrays: list[np.ndarray]
) -> None:
    mock.assert_called_once()
    call_arrays = mock.call_args[0]
    assert len(call_arrays) == len(expected_arrays)
    for expected_array, call_array in zip(expected_arrays, call_arrays):
        assert np.array_equal(expected_array, call_array)


@pytest.fixture
def train_data() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(123)
    return np.random.uniform(size=(10, 10)), np.array(
        [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    )


@pytest.fixture
def prediction_generator() -> PredictionGenerator:
    mock_predictions = [
        np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1]),
        np.array([1, 0, 1, 1, 0]),
        np.array([1, 1]),
    ]
    mock_predictions_generator = (pred for pred in mock_predictions)

    def prediction_generator_side_effect(*args, **kwargs) -> np.ndarray:
        return next(mock_predictions_generator)

    prediction_generator = FixedThresholdPredictionGenerator(0.5)
    prediction_generator.make_prediction_train = Mock(  # type: ignore
        side_effect=prediction_generator_side_effect
    )

    return prediction_generator


@pytest.fixture
def estimator_generator() -> EstimatorGenerator:
    estimators = [
        LogisticRegression(),
        LogisticRegression(),
        LogisticRegression(),
    ]
    estimators[0].fit = Mock(return_value=estimators[0])
    estimators[1].fit = Mock(return_value=estimators[1])
    estimators[2].fit = Mock(return_value=estimators[2])
    return PredefinedEstimatorsGenerator(estimators)


@pytest.fixture
def fitting_history() -> list[np.ndarray]:
    return [
        np.array(
            [False, True, False, True, False, True, True, True, False, False]
        ),
        np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
            ]
        ),
        np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ]
        ),
    ]


@pytest.fixture
def routing_model() -> ClassifierMixin:
    routing_model = LogisticRegression()
    routing_model.fit = Mock(return_value=routing_model)
    return routing_model


@pytest.fixture
def routing_model_fit_data() -> np.ndarray:
    return np.array([0, 1, 0, 1, 0, 1, 2, 2, 0, 0])


def test__fit_estimators(
    train_data: Tuple[np.ndarray, np.ndarray],
    prediction_generator: PredictionGenerator,
    estimator_generator: EstimatorGenerator,
    routing_model: ClassifierMixin,
    fitting_history: list[np.ndarray],
) -> None:
    # Given
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )
    X, y = train_data
    expected_fitting_history = fitting_history

    # When
    estimators, actual_fitting_history = (
        hellsemble._Hellsemble__fit_estimators(X, y)
    )

    # Then
    # Check prediction history
    assert len(expected_fitting_history) == len(actual_fitting_history)
    for (
        expected_fitting_history_entry,
        actual_predictions_history_entry,
    ) in zip(
        expected_fitting_history,
        actual_fitting_history,
    ):
        assert np.array_equal(
            expected_fitting_history_entry,
            actual_predictions_history_entry,
        )
    # Check correct data used during fitting estimators
    assert_called_once_with_numpy_arrays(estimators[0].fit, [X, y])
    assert_called_once_with_numpy_arrays(
        estimators[1].fit,
        [
            X[fitting_history[0]],
            y[fitting_history[0]],
        ],
    )
    assert_called_once_with_numpy_arrays(
        estimators[2].fit,
        [
            X[fitting_history[1]],
            y[fitting_history[1]],
        ],
    )


def test__generate_fitting_data_for_routing_model(
    prediction_generator: PredictionGenerator,
    estimator_generator: EstimatorGenerator,
    routing_model: ClassifierMixin,
    fitting_history: list[np.ndarray],
    routing_model_fit_data: np.ndarray,
) -> None:
    # Given
    expected_routing_model_fit_data = routing_model_fit_data
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )

    # When
    actual_routing_model_fit_data = (
        hellsemble._Hellsemble__generate_fitting_data_for_routing_model(
            fitting_history
        )
    )

    # Then
    assert np.array_equal(
        expected_routing_model_fit_data, actual_routing_model_fit_data
    )


def test__fit_routing_model(
    train_data: Tuple[np.ndarray, np.ndarray],
    prediction_generator: PredictionGenerator,
    estimator_generator: EstimatorGenerator,
    routing_model: ClassifierMixin,
    fitting_history: list[np.ndarray],
    routing_model_fit_data: np.ndarray,
) -> None:
    # Given
    X, _ = train_data
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )

    # When
    actual_routing_model = hellsemble._Hellsemble__fit_routing_model(
        routing_model, X, fitting_history
    )

    # Then
    assert_called_once_with_numpy_arrays(
        actual_routing_model.fit, [X, routing_model_fit_data]
    )


def test__fitting_stop_condition_when_holds(
    prediction_generator: PredictionGenerator,
    estimator_generator: EstimatorGenerator,
    routing_model: ClassifierMixin,
) -> None:
    # Given
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )
    fitting_history = [np.full((100), False)]

    # When / Then
    assert hellsemble._Hellsemble__fitting_stop_condition(fitting_history)


def test__fitting_stop_condition_when_does_not_hold(
    prediction_generator: PredictionGenerator,
    estimator_generator: EstimatorGenerator,
    routing_model: ClassifierMixin,
) -> None:
    # Given
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )
    fitting_history = [np.full((100), False)]
    fitting_history[0][:10] = True

    # When / Then
    assert not hellsemble._Hellsemble__fitting_stop_condition(fitting_history)


def test_predict_proba(
    train_data: Tuple[np.ndarray, np.ndarray],
    prediction_generator: PredictionGenerator,
    estimator_generator: EstimatorGenerator,
) -> None:
    # Given
    X, _ = train_data
    routing_model = LogisticRegression()
    routing_model.predict = Mock(
        return_value=np.array([0, 1, 0, 1, 2, 2, 0, 1, 1, 0])
    )
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )

    class PredictProbaSideEffectWithFixedValue:

        def __init__(self, class_1_proba: float):
            self.class_1_proba = class_1_proba

        def __call__(self, X: np.ndarray) -> np.ndarray:
            return np.repeat(
                [[1 - self.class_1_proba, self.class_1_proba]], X.shape[0], 0
            )

    estimators = [
        LogisticRegression(),
        LogisticRegression(),
        LogisticRegression(),
    ]
    estimators[0].predict_proba = Mock(
        side_effect=PredictProbaSideEffectWithFixedValue(0.1)
    )
    estimators[1].predict_proba = Mock(
        side_effect=PredictProbaSideEffectWithFixedValue(0.2)
    )
    estimators[2].predict_proba = Mock(
        side_effect=PredictProbaSideEffectWithFixedValue(0.3)
    )
    hellsemble.estimators = estimators
    expected_probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.7, 0.3],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )

    # When
    actual_probabilities = hellsemble.predict_proba(X)

    # Then
    assert np.array_equal(expected_probabilities, actual_probabilities)


def test_predict(
    train_data: Tuple[np.ndarray, np.ndarray],
    estimator_generator: EstimatorGenerator,
) -> None:
    # Given
    # Mock routing model
    X, _ = train_data
    routing_model = LogisticRegression()
    routing_model.predict = Mock(
        return_value=np.array([0, 1, 0, 1, 2, 2, 0, 1, 1, 0])
    )
    # Mock make_prediction
    estimators = [
        LogisticRegression(),
        LogisticRegression(),
        LogisticRegression(),
    ]

    def make_prediction_side_effect(estimator: ClassifierMixin, X: np.ndarray):
        prediction_mapping = {id(est): i for i, est in enumerate(estimators)}
        return np.full((X.shape[0]), prediction_mapping[id(estimator)])

    prediction_generator = FixedThresholdPredictionGenerator(0.5)
    prediction_generator.make_prediction = Mock(  # type: ignore
        side_effect=make_prediction_side_effect
    )
    expected_predictions = np.array([0, 1, 0, 1, 2, 2, 0, 1, 1, 0])
    # Mock hellsemble
    hellsemble = Hellsemble(
        estimator_generator, prediction_generator, routing_model
    )
    hellsemble.estimators = estimators

    # When
    actual_predictions = hellsemble.predict(X)

    # Then
    assert np.array_equal(expected_predictions, actual_predictions)
