import numpy as np

from sedgen.sedgen import calculate_volume_sphere


class Bin:
    def __init__(self, n_bins=1500, lower=-10, higher=5):
        self.n_bins = n_bins
        self.lower = lower
        self.higher = higher
        self.range = self.higher - self.lower

        # Size bins
        self.size_bins = self.initialize_bins(self.lower,
                                              self.upper,
                                              self.n_bins+1)
        self.size_bins_medians = calculate_bins_medians(self.size_bins)

        # Volume bins
        self.volume_bins = calculate_volume_sphere(self.size_bins)
        self.volume_bins_medians = calculate_bins_medians(self.volume_bins)

        # Search size bins
        self.search_size_bins = self.initialize_bins(self.lower - self.range,
                                                     self.upper,
                                                     self.n_bins*2-1)
        self.search_size_bins_medians = \
            calculate_bins_medians(self.search_size_bins)
        self.ratio_search_size_bins = \
            calculate_ratio_bins(self.search_size_bins_medians)

        # Search volume bins
        self.search_volume_bins = \
            calculate_volume_sphere(self.search_size_bins)
        self.search_size_bins_medians = \
            calculate_bins_medians(self.search_volume_bins)
        self.ratio_search_volume_bins = \
            calculate_ratio_bins(self.search_volume_bins_medians)


def initialize_bins(lower, upper, n_bins):
    size_bins = np.array([2.0**x for x in np.linspace(lower, upper, n_bins)])
    return size_bins


def calculate_bins_medians(bins):
    bins_medians = bins[..., :-1] + 0.5 * np.diff(bins, axis=-1)
    return bins_medians


def calculate_ratio_bins(bins):
    ratio_bins = bins / bins[..., -1, None]
    return ratio_bins
