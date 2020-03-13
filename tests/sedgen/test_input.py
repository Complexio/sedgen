import pytest
import itertools

import sedgen
print(sedgen.__path__)

import sedgen.input as input_

__author__ = "Bram Paredis"
__copyright__ = "Bram Paredis"
__license__ = "mit"

@pytest.fixture
def fixture_mineral_classes():
    dataset = input_.Input(["Q" ,"P", "K", "B", "O", "A"])

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
        expected = 21

        assert actual == pytest.approx(expected)
