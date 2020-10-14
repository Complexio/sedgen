import numpy as np
import numba as nb
from fast_histogram import histogram1d, histogram2d


def count_interfaces(A):
    """Count number frequencies of crystal interfaces
    https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/16973510"""

    # A = self.interface_pairs

    b = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
    unq_a, unq_cnt = np.unique(b, return_counts=True)
    unq_a = unq_a.view(A.dtype).reshape(-1, A.shape[1])

    return unq_a, unq_cnt

    # Numba does not yet support the return_counts keyword argument of
    # np.unique, unfortunately.


def convert_counted_interfaces_to_matrix(unq_a, unq_cnt, n_minerals):
    """Converts tuple resulting from count_interfaces call to numpy
    matrix. Doesn't break if not all entries of the matrix are present
    in the count.
    """
    count_matrix = np.zeros((n_minerals, n_minerals), dtype=np.int32)

    for index, count in zip(unq_a, unq_cnt):
        count_matrix[tuple(index)] = count

    return count_matrix


def count_and_convert_interfaces_to_matrix(pcg, n_minerals):
    """Counts frequencies of interfaces via fasthistogram module and subsequently converts them to matrix format."""
    return histogram2d(pcg[:-1], pcg[1:],
                       range=[[0, n_minerals], [0, n_minerals]],
                       bins=n_minerals).astype(np.int32)


def count_items(array, n_bins):
    return histogram1d(array, bins=n_bins, range=[0, n_bins])


# To Do: interface strengths matrix still needs to be added here
# instead of interface_proportions_normalized matrix.
def get_interface_strengths_prob(interface_strengths_matrix, interface_array):
    interface_strengths = \
        interface_strengths_matrix[interface_array[:-1],
                                   interface_array[1:]]
    return interface_strengths


def get_interface_size_prob(crystal_size_array):
    interface_size_prob = \
        np.min(create_pairs(crystal_size_array),
               axis=1)
    # min_filter1d_valid_strided(crystal_size_array, 2)
    # Since this represents probabilities, we don't want zeros as a
    # possible value but a 0 size bin exists.
    # Therefore all values are raised by 1 to go around this issue.
    interface_size_prob += 1

    return interface_size_prob


@nb.njit(cache=True, nogil=True)
def calculate_volume_sphere(r, diameter=True):
    """Calculates volume of a sphere

    Parameters:
    -----------
    r : np.array(float)
        Crystal sizes
    diameter : bool(optional)
        Whether given sizes are the diameter (True) or radius (False);
        defaults to True

    Returns:
    --------
    volume : np.array(float)
        Volumes of spheres
    """
    if diameter:
        r = r * 0.5

    volume = 4/3 * r*r*r * np.pi
    return volume


@nb.njit(cache=True, nogil=True)
def calculate_equivalent_circular_diameter(volume):
    """Calculates the equivalent circular diameter based on a given
    volume"""
    diameter = 2 * (3/4 * volume / np.pi) ** (1/3)

    return diameter


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052"""
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return \
        np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def min_filter1d_valid_strided(a, W):
    """https://stackoverflow.com/questions/43288542/max-in-a-sliding-window-in-numpy-array"""
    return strided_app(a, W, S=1).min(axis=1)


@nb.njit(cache=True)
def weighted_bin_count(a, w, ml=None):
    """Returns weighted counts per bin"""
    return np.bincount(a, weights=w, minlength=ml)


@nb.njit(cache=True)
def bin_count(a):
    """Returns counts per bin"""
    return np.bincount(a)


@nb.njit(cache=True)
def normalize(data):
    """Normalize given data so that sum equals 1"""
    return data / np.sum(data)


def create_pairs(data):
    """Creates pairs of crystals, i.e. interfaces"""
    return np.dstack((data[:-1],
                      data[1:]))[0]


def initialize_size_bins(lower=-10, upper=5, n_bins=1500):
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

    bins = [2.0**x for x in np.linspace(lower, upper, n_bins+1)]

    return np.array(bins)


def calculate_bins_medians(bins):
    """Calculates the median value per two neighbouring bin edges,
    i.e. per bin.
    """

    bins_medians = bins[..., :-1] + 0.5 * np.diff(bins, axis=-1)

    return bins_medians


def initialize_search_size_bins(n_bins, lower=-25, upper=5):
    """Initalizes bins used for searching the best fitting bin during
    the intra_cb_dict initalization.
    """
    search_size_bins = [2.0**x for x in np.linspace(lower,
                                                    upper,
                                                    n_bins*2-1)]
    return np.array(search_size_bins)


def calculate_ratio_search_bins(search_bins_medians):
    """Calculates the ratio between all search_bins_medians an the final
    bin's median value."""
    return search_bins_medians / search_bins_medians[-1]


def expand_array(a, expand=1):
    """Expands an 2d array across both axes by adding zero rows and zero
     colums."""
    a_expanded = \
        np.vstack(
            (np.hstack((a, np.zeros((a.shape[0], expand)))),
             np.zeros((expand, a.shape[1]+expand)))
            )
    return a_expanded
