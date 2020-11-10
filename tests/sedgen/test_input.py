import pytest
import itertools
import numpy as np

import sedgen
from sedgen.initialization import SedGen

print(sedgen.__path__)

__author__ = "Bram Paredis"
__copyright__ = "Bram Paredis"
__license__ = "mit"


@pytest.fixture
def fixture_mineral_classes():
    modal_mineralogy_ = np.array([0.30591989, 0.38159713, 0.26209888,
                                  0.01882560, 0.00799247, 0.02356603])
    csd_means_ = np.array([0.309, 0.330, 0.244, 0.223, 0.120, 0.122])
    csd_stds_ = np.array([0.823, 0.683, 0.817, 0.819, 0.554, 0.782])

    dataset = \
        SedGen(minerals=["Q", "P", "K", "B", "O", "A"],
               parent_rock_volume=1e6,
               modal_mineralogy=modal_mineralogy_,
               csd_means=csd_means_,
               csd_stds=csd_stds_,
               interfacial_composition=None,
               learning_rate=100000,
               discretization_init=False)

    yield dataset


class TestGetInterfaceLabels(object):
    def test_correct_labels(self, fixture_mineral_classes):
        actual = fixture_mineral_classes.get_interface_labels()
        expected = ['QQ', 'QP', 'QK', 'QB', 'QO', 'QA',
                          'PP', 'PK', 'PB', 'PO', 'PA',
                                'KK', 'KB', 'KO', 'KA',
                                      'BB', 'BO', 'BA',
                                            'OO', 'OA',
                                                  'AA']

        assert actual == expected

    def test_correct_number_of_labels(self, fixture_mineral_classes):
        actual = len(fixture_mineral_classes.get_interface_labels())
        expected = np.sum(np.arange(fixture_mineral_classes.n_minerals+1))

        assert actual == pytest.approx(expected)
