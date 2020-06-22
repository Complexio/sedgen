import numpy as np
import itertools

class Test():

class Input():
    """Set up initial parameters of dataset

    Parameters:
    -----------
    m : list
        Mineral classes present in dataset
    d : int (optional)
        Number of size classes forming a geometric series.
        Defaults to 10.

    """
    def __init__(self, mineral_labels, d=10):
        self.mineral_labels = mineral_labels
        self.m = len(self.mineral_labels)
        self.d = d

        # Initialize empty arrays for bookkeeping
        # number of crystals forming part of polycrystalline grains (pcg)
        self.Y = np.zeros((m, d))
        # number of crystals occurring as monocrystalline grains (mcg)
        self.Z = np.zeros((m, d))
        # number of polycrystalline grains (pcg)
        self.R = np.zeros((1, d))


    def get_interface_labels(self):
        """Returns list of combinations of interfaces between provided
        list of minerals
        """

        interface_labels = \
            ["".join(pair) for pair in
            itertools.combinations_with_replacement(self.mineral_labels, 2)]

        return interface_labels
