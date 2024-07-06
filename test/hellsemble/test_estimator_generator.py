from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from hellsemble.estimator_generator import PredefinedEstimatorsGenerator


@pytest.fixture
def train_data() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(123)
    return np.random.uniform(size=(10, 10)), (
        np.random.uniform(size=(10)) >= 0.5
    ).astype(int)


def test_predefined_estimators_generator_fits_estimator(
    train_data: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    estimator = LogisticRegression()
    fit_mock = Mock(return_value=estimator)
    estimator.fit = fit_mock
    X, y = train_data
    estimator_generator = PredefinedEstimatorsGenerator([estimator])

    # When
    generated_estimator = estimator_generator.fit_next_estimator(X, y)

    # Then
    assert generated_estimator == estimator
    fit_mock.assert_called_once_with(X, y)


def test_predefined_estimators_generator_returns_proper_sequence_of_estimators(
    train_data: Tuple[np.ndarray, np.ndarray],
) -> None:
    # Given
    estimator1 = LogisticRegression()
    estimator2 = LogisticRegression()
    X, y = train_data
    estimator_generator = PredefinedEstimatorsGenerator(
        [estimator1, estimator2]
    )

    # When
    generated_estimator1 = estimator_generator.fit_next_estimator(X, y)
    generated_estimator2 = estimator_generator.fit_next_estimator(X, y)

    # Then
    assert generated_estimator1 == estimator1
    assert generated_estimator2 == estimator2


def test_predefined_estimators_generator_fit_next_estimators_after_all_estimators_used(  # noqa: E501
    train_data: Tuple[np.ndarray, np.ndarray],
) -> None:
    # Given
    estimator_generator = PredefinedEstimatorsGenerator(
        [LogisticRegression(), LogisticRegression()]
    )
    X, y = train_data

    # When
    estimator_generator.fit_next_estimator(X, y)
    estimator_generator.fit_next_estimator(X, y)

    # Then
    assert not estimator_generator.has_next()
    with pytest.raises(StopIteration):
        estimator_generator.fit_next_estimator(X, y)
