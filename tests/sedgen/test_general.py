import pytest
import numpy as np

import sedgen
import sedgen.general as gen

print(sedgen.__path__)

__author__ = "Bram Paredis"
__copyright__ = "Bram Paredis"
__license__ = "mit"

@pytest.fixture
def fixture_crystal_diameters():
    crystal_diameters = np.array([0.2, 20, 0.05])

    yield crystal_diameters

@pytest.fixture
def fixture_crystal_volumes():
    crystal_volumes = \
        np.array([4.18879020e-03, 4.18879020e+03, 6.54498469e-05])

    yield crystal_volumes

@pytest.fixture
def fixture_modal_mineralogy():
    modal_mineralogy = np.array([0.28, 0.35, 0.24, 0.02, 0.01, 0.03])
    yield modal_mineralogy

@pytest.fixture
def fixture_zero_array():
    zero_array = np.zeros((3, 3))

    yield zero_array


class TestCalculateVolumeSphere(object):
    def test_correct_outcome_of_diameters(self, fixture_crystal_diameters):
        actual = gen.calculate_volume_sphere(fixture_crystal_diameters)
        expected = np.array([4.18879020e-03, 4.18879020e+03, 6.54498469e-05])

        assert actual == pytest.approx(expected)

    def test_correct_outcome_of_radii(self, fixture_crystal_diameters):
        radii = fixture_crystal_diameters / 2
        actual = gen.calculate_volume_sphere(fixture_crystal_diameters,
                                             diameter=False)
        expected = \
            np.array([4.18879020e-03, 4.18879020e+03, 6.54498469e-05]) * 8

        assert actual == pytest.approx(expected)


class TestCalculateEquivalentCircularDiameter(object):
    def test_correct_outcome(self, fixture_crystal_volumes):
        actual = \
            gen.calculate_equivalent_circular_diameter(fixture_crystal_volumes)
        expected = np.array([0.2, 20, 0.05])

        assert actual == pytest.approx(expected)


class TestNormalize(object):
    def test_correct_outcome(self, fixture_modal_mineralogy):
        actual = gen.normalize(fixture_modal_mineralogy)
        expected = np.array([0.30107527, 0.37634409, 0.25806452,
                             0.02150538, 0.01075269, 0.03225806])

        assert actual == pytest.approx(expected)


class TestExpandArray(object):
    def test_one_expansion(self, fixture_zero_array):
        actual = gen.expand_array(fixture_zero_array).shape
        expected = (fixture_zero_array.shape[0] + 1,
                    fixture_zero_array.shape[1] + 1)

        assert actual == expected

    def test_two_expansions(self, fixture_zero_array):
        actual = gen.expand_array(fixture_zero_array, 2).shape
        expected = (fixture_zero_array.shape[0] + 2,
                    fixture_zero_array.shape[1] + 2)

        assert actual == expected
