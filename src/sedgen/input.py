import numpy as np
from scipy.stats import lognorm
import itertools

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
        self.Y = np.zeros((self.m, self.d))
        # number of crystals occurring as monocrystalline grains (mcg)
        self.Z = np.zeros((self.m, self.d))
        # number of polycrystalline grains (pcg)
        self.R = np.zeros((1, self.d))


    def get_interface_labels(self):
        """Returns list of combinations of interfaces between provided
        list of minerals

        """

        interface_labels = \
            ["".join(pair) for pair in
            itertools.combinations_with_replacement(self.mineral_labels, 2)]

        return interface_labels


    def get_number_of_crystals(self):
        """Returns total number of crystals (mcg + pcg) currently
        present in the system per grain size class

        """

        X = self.Y + self.Z

        return X


    def set_random_grain_size_distributions(self):
        """Create random crystal size distributions for all present
        mineral classes

        """
        minerals_csd = []

        prng = np.random.RandomState(1234)
        means = prng.normal(0, 1, self.m)
        stds = prng.lognormal(0, 1, self.m)

        for i, mineral in enumerate(self.mineral_labels):
            minerals_csd.append(lognorm(means[i], stds[i]))

        return np.array(minerals_csd)


    def set_grain_size_distributions_in_bins(self, y_phi):
        """Put created random crystal size probability distributions
        in bins

        """

        Y = np.zeros((self.m, self.d))

        for i, csd_mineral in enumerate(y_phi):
            Y[i] = np.histogram(csd_mineral, bins=self.d)

        return Y


    def get_random_csd_sample(self, minerals_csd, sample_size=1000):
        """Get a random sample of specified size for all mineral classes
        in the system according to randomly set crystam size probability
        distributions.

        """

        y = lambda x: x.rvs(sample_size) for minerals_csdsample_size)

        return y


    def convert_to_phi_scale(self, y):
        """Calculate phi scale for random sample

        """

        y_phi = calc_phi(y)

        return y_phi


    def calc_phi(self, diam):
        """Convert diameter in mm to Phi scale.

        """

        if np.exp(diam) != 0:
            phi = -(np.log10(np.exp(diam)) / np.log10(2))
        else:
            phi = 99

        return phi
