# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

import sedgen
print(sedgen.__path__)

import sedgen.geostatistics as geostat

__author__ = "Bram Paredis"
__copyright__ = "Bram Paredis"
__license__ = "mit"


@pytest.fixture
def fixture_df():
    test_df = pd.DataFrame([[0.2, 0.3, 0.5],
                            [0.4, 0.4, 0.2],
                            [0.1, 0.6, 0.3]])

    yield test_df


@pytest.fixture
def fixture_array():
    test_array = np.array([0.2, 0.3, 0.5, 0.4, 0.4, 0.2, 0.1, 0.6, 0.3])

    yield test_array


class TestClr(object):

    def test_clr(self, fixture_df):
        actual = geostat.clr(fixture_df).values
        expected = np.array([[-0.44058528, -0.03512017,  0.47570545],
                             [ 0.23104906,  0.23104906, -0.46209812],
                             [-0.96345725,  0.82830222,  0.13515504]])

        assert actual == pytest.approx(expected)

    def test_clr_for_zero_sum(self, fixture_df):
        actual = np.sum(geostat.clr(fixture_df).values, axis=1)
        expected = 0.0

        assert actual == pytest.approx(expected)


class TestAlr(object):

    def test_alr(self, fixture_df):
        actual = geostat.alr(fixture_df).values
        expected = np.array([[0.0, 0.405465,  0.916291],
                             [0.0, 0.000000, -0.693147],
                             [0.0, 1.791759,  1.098612]])

        assert actual == pytest.approx(expected)

    def test_alr_for_zero_column(self, fixture_df):
        actual = geostat.alr(fixture_df).values[:, 0]
        expected = 0.0

        assert actual == pytest.approx(expected)


class TestGeometrics(object):

    def test_geo_mean(self, fixture_df):
        actual = geostat.geometrics(fixture_df)[0]
        expected = np.array([0.2, 0.41601676, 0.31072325])

        assert actual == pytest.approx(expected)

    def test_geo_std(self, fixture_df):
        actual = geostat.geometrics(fixture_df)[1].values
        expected = np.array([1.76112411, 1.32887762, 1.45484234])

        assert actual == pytest.approx(expected)
