#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Lovkush Agarwal", "Markus Löning"]
__all__ = [
    "test_factory_method_recursive",
    "test_factory_method_direct",
    "test_factory_method_ts_direct",
    "test_factory_method_ts_recursive",
    "test_multioutput_direct_tabular",
    "test_sliding_window_tranform_tabular",
    "test_sliding_window_tranform_panel",
]

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose._reduce import DirectRegressionForecaster
from sktime.forecasting.compose._reduce import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import MultioutputRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import ReducedForecaster
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.tests._config import TEST_WINDOW_LENGTHS
from sktime.transformations.panel.reduce import Tabularizer
from sktime.utils.validation.forecasting import check_fh

N_TIMEPOINTS = [13, 17]
N_VARIABLES = [1, 3]
FH = ForecastingHorizon(1)


def _make_y_X(n_timepoints, n_variables):
    # We generate y and X values so that the y should always be greater
    # than its lagged values and the lagged and contemporaneous values of the
    # exogenous variables X
    assert n_variables < 10

    base = np.arange(n_timepoints)
    y = pd.Series(base + n_variables / 10)

    if n_variables > 1:
        X = np.column_stack([base + i / 10 for i in range(1, n_variables)])
        X = pd.DataFrame(X)
    else:
        X = None

    return y, X


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_variables", N_VARIABLES)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_sliding_window_tranform_tabular(n_timepoints, window_length, n_variables, fh):
    y, X = _make_y_X(n_timepoints=n_timepoints, n_variables=n_variables)
    fh = check_fh(fh, enforce_relative=True)
    fh_max = fh[-1]
    effective_window_length = window_length + fh_max - 1

    yt, Xt = _sliding_window_transform(
        y, window_length=window_length, fh=fh, X=X, return_panel=False
    )
    assert yt.shape == (n_timepoints - effective_window_length, len(fh))

    # Check y values for first step in fh.
    first_step = yt[:, [0]]
    expected = y[np.arange(window_length, n_timepoints - fh_max + 1)]
    np.testing.assert_array_equal(first_step.ravel(), expected)

    # The transformed Xt array contains lagged values for each variable, plus the
    # contemporaneous values for the exogenous variables.
    assert Xt.shape == (yt.shape[0], (window_length * n_variables) + n_variables - 1)
    assert np.all(Xt < first_step)


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_variables", N_VARIABLES)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_sliding_window_tranform_panel(n_timepoints, window_length, n_variables, fh):
    y, X = _make_y_X(n_timepoints=n_timepoints, n_variables=n_variables)
    fh = check_fh(fh, enforce_relative=True)
    fh_max = fh[-1]
    effective_window_length = window_length + fh_max - 1

    yt, Xt = _sliding_window_transform(
        y, window_length=window_length, X=X, fh=fh, return_panel=True
    )
    assert yt.shape == (n_timepoints - effective_window_length, len(fh))

    # Check y values.
    actual = yt[:, 0]
    expected = y[np.arange(window_length, n_timepoints - fh_max + 1)]
    np.testing.assert_array_equal(actual, expected)

    # Given the input data, all of the value in the transformed Xt array should be
    # smaller than the transformed yt target array.
    assert Xt.shape == (yt.shape[0], n_variables, window_length)
    assert np.all(Xt < yt[:, np.newaxis, [0]])


def test_factory_method_recursive():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = ReducedForecaster(regressor, scitype="regressor", strategy="recursive")
    f2 = RecursiveRegressionForecaster(regressor)

    actual = f1.fit(y_train).predict(fh)
    expected = f2.fit(y_train).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_factory_method_direct():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = ReducedForecaster(regressor, scitype="regressor", strategy="direct")
    f2 = DirectRegressionForecaster(regressor)

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = f2.fit(y_train, fh=fh).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_factory_method_ts_recursive():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = ReducedForecaster(ts_regressor, scitype="ts_regressor", strategy="recursive")
    f2 = RecursiveTimeSeriesRegressionForecaster(ts_regressor)

    actual = f1.fit(y_train).predict(fh)
    expected = f2.fit(y_train).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_factory_method_ts_direct():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = ReducedForecaster(ts_regressor, scitype="ts_regressor", strategy="direct")
    f2 = DirectTimeSeriesRegressionForecaster(ts_regressor)

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = f2.fit(y_train, fh=fh).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_multioutput_direct_tabular():
    # multioutput and direct strategies with linear regression
    # regressor should produce same predictions
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = MultioutputRegressionForecaster(regressor)
    f2 = DirectRegressionForecaster(regressor)

    preds1 = f1.fit(y_train, fh=fh).predict(fh)
    preds2 = f2.fit(y_train, fh=fh).predict(fh)

    # assert_almost_equal does not seem to work with pd.Series objects
    np.testing.assert_almost_equal(preds1.to_numpy(), preds2.to_numpy(), decimal=5)
