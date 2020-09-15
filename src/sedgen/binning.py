import numpy as np


class Bin:
    def __init__(self, n_bins=1500, lower=-10, higher=5):
        self.n_bins = n_bins
        self.lower = lower
        self.higher = higher
        self.range = self.higher - self.lower

        self.size_bins = self.initialize_size_bins()
        self.size_bins_medians = self.calculate_bins_medians()

        self.volume_bins = self.initialize_volume_bins()
        self.volume_bins_medians = \
            self.calculate_bins_medians(self.volume_bins)

        self.search_bins = self.initialize_search_bins()
        self.search_bins_medians = self.calculate_search_bins_medians()
        self.ratio_search_bins = self.calculate_ratio_search_bins()

    def initialize_size_bins(self):
        size_bins = \
            [2.0**x for x
             in np.linspace(self.lower, self.upper, self.n_bins+1)]
        return np.array(size_bins)

    def calculate_bins_medians(self):
        bins_medians = np.array([(self.size_bins[i] + self.size_bins[i+1]) / 2
                                 for i in range(len(self.size_bins) - 1)])
        return bins_medians

    def initialize_volume_bins(self):
        return self.calculate_volume_sphere(self.size_bins)

    def initialize_search_bins(self):
        search_bins = [2.0**x for x in np.linspace(self.lower - self.range,
                                                   self.higher,
                                                   self.n_bins*2-1)]
        return self.calculate_volume_sphere(np.array(search_bins))

    def calculate_search_bins_medians(self):
        return self.calculate_bins_medians(self.search_bins)

    def calculate_ratio_search_bins(self):
        return self.search_bins_medians / self.search_bins_medians[-1]
