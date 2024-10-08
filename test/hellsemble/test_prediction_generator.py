from unittest.mock import Mock

import numpy as np

from hellsemble.prediction_generator import FixedThresholdPredictionGenerator


def test_fixed_threshold_prediction_generator_make_prediction_train() -> None:
    # Given
    estimator = Mock()
    estimator.predict_proba = Mock(
        return_value=np.array([[0.2, 0.8], [0.8, 0.2]])
    )
    prediction_generator = FixedThresholdPredictionGenerator(0.5)
    expected_prediction = np.array([1, 0])

    # When
    actual_prediction = prediction_generator.make_prediction_train(
        estimator, np.array([])
    )

    # Then
    assert np.array_equal(expected_prediction, actual_prediction)


def test_fixed_threshold_prediction_generator_make_prediction() -> None:
    # Given
    estimator = Mock()
    estimator.predict_proba = Mock(
        return_value=np.array([[0.2, 0.8], [0.8, 0.2]])
    )
    prediction_generator = FixedThresholdPredictionGenerator(0.5)
    expected_prediction = np.array([1, 0])

    # When
    actual_prediction = prediction_generator.make_prediction(
        estimator, np.array([])
    )

    # Then
    assert np.array_equal(expected_prediction, actual_prediction)
