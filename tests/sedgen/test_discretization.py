import pytest
import numpy as np

import sedgen
import sedgen.discretization as discr

print(sedgen.__path__)

__author__ = "Bram Paredis"
__copyright__ = "Bram Paredis"
__license__ = "mit"


@pytest.fixture
def fixture_default_bin_edges():
    default_bin_edges = \
        np.array([2.0**x for x in np.linspace(-10, 5, 1501)])

    yield default_bin_edges


@pytest.fixture
def fixture_custom_bin_edges():
    custom_bin_edges = np.array([0.5, 0.79370053, 1.25992105, 2.])

    yield custom_bin_edges


class TestInitializeBins(object):
    def test_bins_with_default_parameters(self, fixture_default_bin_edges):
        actual = discr.initialize_bins()
        expected = fixture_default_bin_edges

        assert actual == pytest.approx(expected)

    def test_bins_with_custom_parameters(self, fixture_custom_bin_edges):
        actual = discr.initialize_bins(-1, 1, 4)
        expected = fixture_custom_bin_edges

        assert actual == pytest.approx(expected)


class TestCalculateBinsMedians(object):
    def test_medians_with_defaults_bins(self, fixture_default_bin_edges):
        actual = discr.calculate_bins_medians(fixture_default_bin_edges)
        expected = fixture_default_bin_edges[..., :-1] \
            + 0.5 * np.diff(fixture_default_bin_edges, axis=-1)

        assert actual == pytest.approx(expected)

    def test_medians_with_custom_bins(self, fixture_custom_bin_edges):
        actual = discr.calculate_bins_medians(fixture_custom_bin_edges)
        expected = fixture_custom_bin_edges[..., :-1] \
            + 0.5 * np.diff(fixture_custom_bin_edges, axis=-1)

        assert actual == pytest.approx(expected)


class TestCalculateRatioBins(object):
    def test_ratios_with_default_bins(self, fixture_default_bin_edges):
        actual = discr.calculate_ratio_bins(fixture_default_bin_edges)
        expected = fixture_default_bin_edges \
            / fixture_default_bin_edges[..., -1, None]

        assert actual == pytest.approx(expected)

    def test_ratios_with_custom_bins(self, fixture_custom_bin_edges):
        actual = discr.calculate_ratio_bins(fixture_custom_bin_edges)
        expected = fixture_custom_bin_edges \
            / fixture_custom_bin_edges[..., -1, None]

        assert actual == pytest.approx(expected)
