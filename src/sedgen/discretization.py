import numpy as np
import numba as nb

from sedgen.general import calculate_volume_sphere


class Bins:
    def __init__(self, lower=-10, upper=5, n_bins=1500):
        self.n_bins = n_bins
        self.n_bin_edges = self.n_bins + 1
        self.lower_bin_edge = lower
        self.upper_bin_edge = upper
        self.bin_range = self.upper_bin_edge - self.lower_bin_edge

        # Size bins
        self.size_bins = initialize_bins(self.lower_bin_edge,
                                         self.upper_bin_edge,
                                         self.n_bin_edges)
        self.size_bins_medians = calculate_bins_medians(self.size_bins)

        # Volume bins
        self.volume_bins = calculate_volume_sphere(self.size_bins)
        self.volume_bins_medians = calculate_bins_medians(self.volume_bins)

        # Search size bins
        self.search_size_bins = \
            initialize_bins(self.lower_bin_edge - self.bin_range,
                            self.upper_bin_edge,
                            self.n_bin_edges*2-1)
        self.search_size_bins_medians = \
            calculate_bins_medians(self.search_size_bins)
        self.ratio_search_size_bins = \
            calculate_ratio_bins(self.search_size_bins_medians)

        # Search volume bins
        self.search_volume_bins = \
            calculate_volume_sphere(self.search_size_bins)
        self.search_volume_bins_medians = \
            calculate_bins_medians(self.search_volume_bins)
        self.ratio_search_volume_bins = \
            calculate_ratio_bins(self.search_volume_bins_medians)


def initialize_bins(lower=-10, upper=5, n_bin_edges=1501):
    """Initializes n_bins size bins for binning of crystal sizes
    through a geometric series of [2**lower, ..., 2**upper]

    Parameters:
    -----------
    lower : float(optional)
        Lower value of exponent; defaults to -10 which equals to a
        crystal size of ca. 0.001 mm
    upper : float(optional)
        Upper value of exponent; defaults to 5 which equals to a crystal
        size of 32 mm
    n_bins : int(optional)
        Number of size bins to use; defaults to 1500. Note that returned
        values will be of size n_bins+1 as the bin_edges are returned.

    Returns:
    --------
    bins : np.array
        Bin edges of size n_bins+1
    """

    bins = np.array([2.0**x for x in np.linspace(lower, upper, n_bin_edges)])

    return bins


def calculate_bins_medians(bins):
    """Calculates the median value per two neighbouring bin edges,
    i.e. per bin.
    """

    bins_medians = bins[..., :-1] + 0.5 * np.diff(bins, axis=-1)

    return bins_medians


def calculate_ratio_bins(bins):
    """Calculates the ratio between all bins and the final bin's value.
    """

    ratio_bins = bins / bins[..., -1, None]

    return ratio_bins


class BinsMatricesMixin():
    def __init__(self):
        # Create bin matrices to capture chemical weathering
        self.size_bins_matrix = \
            self.create_bins_matrix(self.size_bins)
        self.volume_bins_matrix = \
            calculate_volume_sphere(self.size_bins_matrix)

        self.size_bins_medians_matrix = \
            calculate_bins_medians(self.size_bins_matrix)
        self.volume_bins_medians_matrix = \
            calculate_bins_medians(self.volume_bins_matrix)

        # Volume change matrix
        self.volume_change_matrix = -np.diff(self.volume_bins_medians_matrix,
                                             axis=0)

        # Negative bin array thresholds
        self.negative_volume_thresholds = \
            np.argmax(self.size_bins_medians_matrix > 0, axis=2)

        # Create search_bins_matrix
        self.search_size_bins_matrix = \
            self.create_bins_matrix(self.search_size_bins)
        self.search_volume_bins_matrix = \
            calculate_volume_sphere(self.search_size_bins_matrix)

        # Create search_bins_medians_matrix
        self.search_size_bins_medians_matrix = \
            calculate_bins_medians(self.search_size_bins_matrix)
        self.search_volume_bins_medians_matrix = \
            calculate_bins_medians(self.search_volume_bins_matrix)

        # Create ratio_search_bins_matrix
        self.ratio_search_size_bins_matrix = calculate_ratio_bins(
            self.search_size_bins_medians_matrix)
        self.ratio_search_volume_bins_matrix = calculate_ratio_bins(
            self.search_volume_bins_medians_matrix)

    def create_bins_matrix(self, bins):
        """Create the matrix holding the arrays with bins which each
        represent the inital bin array minus x times the chemical
        weathering rate per mineral class.
        """

        bins_matrix = \
            np.array([[bins - n * self.chem_weath_rates[m]
                      for m in range(self.pr_n_minerals)]
                     for n in range(self.n_steps)])

        return bins_matrix


class McgBreakPatternMixin():
    def __init__(self):
        # Determine intra-crystal breakage discretization 'rules'
        self.intra_cb_dict, self.intra_cb_breaks, self.diffs_volumes = \
            determine_intra_cb_dict(self.n_bins * 2 - 2,
                                    self.ratio_search_volume_bins)

        self.intra_cb_dict_keys = \
            np.array(list(self.intra_cb_dict.keys()))
        self.intra_cb_dict_values = \
            np.array(list(self.intra_cb_dict.values()))

        # Create array with corresponding bins to intra_cb_thesholds
        # for matrix of bin arrays
        self.intra_cb_threshold_bin_matrix = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype=np.uint16)
        for n in range(self.n_steps):
            for m in range(self.pr_n_minerals):
                self.intra_cb_threshold_bin_matrix[n, m] = \
                    np.argmax(self.size_bins_medians_matrix[n, m] >
                              self.intra_cb_thresholds[m])

        # Create intra_cb_dicts for all bin_arrays
        self.intra_cb_breaks_matrix, \
            self.diffs_volumes_matrix = self.create_intra_cb_dicts_matrix()

    def create_intra_cb_dicts_matrix(self):
        # Need to account for 'destruction' of geometric series due to
        # chemical weathering --> implement chemical weathering rates
        # into the function somehow.
        intra_cb_breaks_matrix = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype='object')
        diffs_volumes_matrix = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype='object')

        for n in range(self.n_steps):
            print(f"{n+1}/{self.n_steps}", end="\r", flush=True)
            for m in range(self.pr_n_minerals):
                intra_cb_breaks_array = \
                    np.zeros(
                        (self.n_bins -
                         self.intra_cb_threshold_bin_matrix[n, m],
                         len(self.intra_cb_breaks)),
                        dtype=np.uint16)
                diffs_volumes_array = \
                    np.zeros(
                        (self.n_bins -
                         self.intra_cb_threshold_bin_matrix[n, m],
                         len(self.intra_cb_breaks)),
                        dtype=np.float64)

                for i, b in \
                    enumerate(range(self.intra_cb_threshold_bin_matrix[n, m] +
                                    self.n_bins,
                                    self.n_bins*2)):
                    intra_cb_breaks_array[i], diffs_volumes_array[i] = \
                        determine_intra_cb_dict_array_version(
                            b, self.ratio_search_volume_bins_matrix[n, m],
                            max_n_values=len(self.intra_cb_breaks))

                intra_cb_breaks_matrix[n, m] = intra_cb_breaks_array
                diffs_volumes_matrix[n, m] = diffs_volumes_array

        return intra_cb_breaks_matrix, diffs_volumes_matrix

# Deprecated?
def determine_intra_cb_dict(bin_label, ratio_search_bins, verbose=False,
                            corr=1, return_arrays=True, max_n_values=None):
    """Determines the relations for the intra-crystal breakage
    discretization.

    Parameters:
    -----------
    bin_label : int
    ratio_search_bins : np.array
    verbose : bool (optional)
        Whether to show verbose information; defaults to False
    corr : int (optional)
        Correction for the found bin; defaults to 1

    Returns:
    --------
    intra_cb_dict: dict
        Dictionary of bin labels for part 1 of broken mcg (keys) and
        part 2 (values)
    diffs : np.array
        Absolute differences in bin label count between part 1 and part 2
        of broken mcg
    diffs_volumes : np.array
        Relative differences in volume between the original mcg and the
        sum of part 1 and part 2 of broken mcg. This thus represents the
        percentage of the formed intra_cb residue with regard to the
        original mcg's volume.

    Example:
    --------
    original mcg label = 1156
    new mcg part 1 label = 1075
    new mcg part 2 label = 1146

    diffs = 1146 - 1075 = 71
    diffs_volumes = 1 - (ratio_bins[1075] + ratio_bins[1146])
    formed intra_cb_residue = bins[1156] * diffs_volumes
    """
    intra_cb_dict = {}
    if max_n_values:
        diffs = np.zeros(max_n_values, dtype=np.uint16)
        diffs_volumes = np.zeros(max_n_values, dtype=np.float64)
    else:
        diffs = []
        diffs_volumes = []

    specific_ratios = \
        ratio_search_bins[:bin_label] / ratio_search_bins[bin_label]

    for i, bin1 in enumerate(range(bin_label-1, 0, -1)):
        bin2_ratio = 1 - specific_ratios[bin1]
        # Minus 1 for found bin so that volume sum of two new mcg
        # is bit less than 100%; remainder goes to residue later on.
        # bin2 = np.argmax(bin2_ratio < specific_ratios) - corr
        bin2 = find_closest(bin2_ratio, specific_ratios, corr=corr)
        if bin2 == -1:
            break

        intra_cb_dict[bin1] = bin2
        if max_n_values:
            diffs[i] = bin1 - bin2
            diffs_volumes[i] = \
                bin2_ratio - specific_ratios[bin2]
        else:
            diffs.append(bin1 - bin2)
            diffs_volumes.append(bin2_ratio - specific_ratios[bin2])

        if (bin1 - bin2 <= corr):
            break

    return intra_cb_dict, diffs, diffs_volumes


@nb.njit(cache=True)
def determine_intra_cb_dict_array_version(bin_label, ratio_search_bins,
                                          max_n_values, corr=1):

    diffs = np.zeros(max_n_values, dtype=np.uint16)
    diffs_volumes = np.zeros(max_n_values, dtype=np.float64)

    specific_ratios = \
        ratio_search_bins[:bin_label] / ratio_search_bins[bin_label]

    for i, bin1 in enumerate(range(bin_label-1, 0, -1)):
        bin2_ratio = 1 - specific_ratios[bin1]
        # Minus 1 for found bin so that volume sum of two new mcg
        # is bit less than 100%; remainder goes to residue later on.
        bin2 = find_closest(bin2_ratio, specific_ratios, corr=corr)
        if bin2 == -1:
            break

        diffs[i] = bin1 - bin2
        diffs_volumes[i] = bin2_ratio - specific_ratios[bin2]

        if (diffs[i] <= corr):
            break

    return diffs, diffs_volumes


@nb.njit(cache=True)
def find_closest(value, lookup, corr=0):
    return np.argmax(value < lookup) - corr
