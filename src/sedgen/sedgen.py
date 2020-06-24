import numpy as np
import numba as nb
import pandas as pd
import itertools
from scipy.stats import truncnorm
import time


"""To Do:
    - Provide option to specify generated minerals based on a number
      instead of filling up a given volume. In the former case, the
      simulated volume attribute also has more meaning.

    - Make sure the differences between the initial minerals_N
      simulation and the later filling of the interfaces and crystal
      size arrays are not too big. Could these differences be avoided to
      start with and thus work with only one representation?

      --> Deviations due to randomness of creating the interface array
          Not sure if this could be solved and if not better to go back
          to old solution as this also provides a way of deactivating
          one of the two parts of the operation. Once we know how many
          crystals per mineral are in a certain volume of parent rock
          for given csd information, we do not need to calculate this
          every time we want to initalize the interface/mineral and
          crystal size arrays.

      --> It also seems that this proposed approach is not really worth
          it time-efficiency wise.

      --> Solution found but still needs to be checked. There remains a
          small offset between the firstly initialized crystal size
          array and the secondly interface array, though. This could be
          resolved by setting the crystal size on 'missing' mineral
          locations in the interface array, by starting back at the
          beginning of the crystal size array.

    - Provide user with option to bin crystal sizes or not. Or make the
      decision automatically based on the total number of crystals
      present.
"""


class SedGen():

    def __init__(self, minerals, parent_rock_volume, modal_mineralogy,
                 csd_means, csd_stds, interfacial_composition=None, learning_rate=1000, timed=False):
        print("---SedGen model initialization started---\n")

        print("Initializing modal mineralogy...")
        self.minerals = minerals
        self.parent_rock_volume = parent_rock_volume
        self.modal_mineralogy = modal_mineralogy

        # Assert that modal mineralogy proportions sum up to unity.
        assert np.isclose(np.sum(modal_mineralogy), 1.0), \
        "Modal mineralogy proportions do not sum to 1"

        self.modal_volume = self.parent_rock_volume * self.modal_mineralogy

        print("Initializing csds...")
        self.csd_means = csd_means
        self.csd_stds = csd_stds

        self.csds = np.array([self.initialize_csd(m)
                              for m in range(len(self.minerals))])

        self.bins = self.initialize_size_bins()

        print("Simulating mineral occurences...", end=" ")
        if timed:
            tic0 = time.perf_counter()

        self.minerals_N, self.simulated_volume, crystal_sizes_per_mineral = \
            self.create_N_crystals(learning_rate=learning_rate)
        if timed:
            toc0 = time.perf_counter()
            print(f" Done in{toc0 - tic0: 1.4f} seconds")
        else:
            print("")

        print("Initializing interfaces...")
        self.interfaces = self.get_interface_labels()
        self.interfacial_composition = interfacial_composition

        self.number_proportions = self.calculate_number_proportions()
        self.interface_proportions = self.calculate_interface_proportions()
        self.interface_proportions_normalized = \
            self.calculate_interface_proportions_normalized()

        transitions_per_mineral = self.create_transitions()
        self.interface_array = \
            create_interface_array(self.minerals_N, transitions_per_mineral)

        self.interface_pairs = create_pairs(self.interface_array)

        if timed:
            print("Counting interfaces...", end=" ")
            tic1 = time.perf_counter()
        else:
            print("Counting interfaces...")
        self.interface_counts = self.count_interfaces()
        self.interface_counts_matrix = \
            self.convert_counted_interfaces_to_matrix()
        if timed:
            toc1 = time.perf_counter()
            print(f"Done in{toc1 - tic1: 1.4f} seconds")

        if timed:
            print("Initializing crystal size array...", end=" ")
            tic2 = time.perf_counter()
        else:
            print("Initializing crystal size array...")
        self.minerals_N_actual = self.calculate_actual_minerals_N()
        # crystal_sizes_per_mineral = self.create_binned_crystal_size_arrays()
        print(self.interface_array.shape)
        print(self.minerals_N)
        print(np.sum(self.minerals_N))
        print([x.shape[0] for x in crystal_sizes_per_mineral])
        print(self.minerals_N_actual)
        self.crystal_size_array = \
            self.fill_main_cystal_size_array(crystal_sizes_per_mineral)
        if timed:
            toc2 = time.perf_counter()
            print(f"Done in{toc2 - tic2: 1.4f} seconds")

        print("Initializing inter-crystal breakage probability arrays...")
        self.interface_location_prob = self.create_interface_location_prob()
        self.interface_strengths_prob = self.get_interface_stregths_prob()
        self.interface_size_prob = self.get_interface_size_prob()

        print("\n---SedGen model initialization finished succesfully---")

    def get_interface_labels(self):
        """Returns list of combinations of interfaces between provided
        list of minerals

        """

        interface_labels = \
            ["".join(pair) for pair in
             itertools.combinations_with_replacement(self.minerals, 2)]

        return interface_labels

    def initialize_csd(self, m, trunc_left=-np.inf, trunc_right=30):
        mean = np.log(self.csd_means[m])
        std = np.exp(self.csd_stds[m])
        if np.isinf(trunc_left):
            pass
        else:
            trunc_left = np.log(trunc_left)
        trunc_right = np.log(trunc_right)
        a, b = (trunc_left - mean) / std, (trunc_right - mean) / std
        return truncnorm(loc=mean, scale=std, a=a, b=b)

    def initialize_size_bins(self, lower=-17.5, upper=5, n_bins=1500):
        bins = [2.0**x for x in np.linspace(lower, upper, n_bins+1)]
        return bins

    def calculate_bins_medians(self):
        bins_medians = [(self.bins[i] + self.bins[i+1]) / 2
                        for i in range(len(self.bins) - 1)]
        return bins_medians

    def calculate_N_crystals(self, m, learning_rate=1000):
        total_volume_mineral = 0
        requested_volume = self.modal_volume[m]
        crystals = np.array([])
        crystals_total = []

        rs = 0
        while total_volume_mineral < requested_volume:
            diff = requested_volume - total_volume_mineral
            crystals_requested = \
                int(diff / (self.modal_mineralogy[m] * learning_rate)) + 1

            crystals_total.append(crystals_requested)
            crystals_to_add = \
                np.exp(self.csds[m].rvs(size=crystals_requested,
                                        random_state=rs))

            crystals = np.append(crystals, crystals_to_add)

            total_volume_mineral += \
                np.sum(calculate_volume_sphere(crystals_to_add,
                                               diameter=True))

            rs += 1
        crystals_binned = \
            np.array(
                pd.cut(crystals,
                       bins=self.bins,
                       labels=range(len(self.bins)-1),
                       right=False),
                dtype=np.uint16)

        return crystals_total, np.sum(crystals_total), \
            total_volume_mineral, crystals_binned

    def create_N_crystals(self, learning_rate=1000):
        minerals_N = {}
        print("|", end="")
        for m, n in enumerate(self.minerals):
            print(n, end="|")
            minerals_N[n] = \
                self.calculate_N_crystals(m=m, learning_rate=learning_rate)
        minerals_N_total = np.array([N[1] for N in minerals_N.values()])
        simulated_volume = np.array([N[2] for N in minerals_N.values()])
        crystals = np.array([N[3] for N in minerals_N.values()])

        return minerals_N_total, simulated_volume, crystals

    def calculate_number_proportions(self):
        return normalize(self.minerals_N).reshape(-1, 1)

    # To Do: add alpha factor to function to handle non-random interfaces
    def calculate_interface_proportions(self):
        interface_proportions_pred = \
            self.number_proportions * self.number_proportions.T

        return interface_proportions_pred

    def calculate_interface_proportions_normalized(self):
        interface_proportions_normalized = \
            np.divide(self.interface_proportions,
                      np.sum(self.interface_proportions, axis=1).reshape(-1, 1)
                      )
        return interface_proportions_normalized

    def create_transitions(self, random_seed=525):
        possibilities = np.arange(len(self.minerals), dtype=np.uint8)

        prng = np.random.RandomState(random_seed)

        transitions_per_mineral = []

        for i, mineral in enumerate(self.minerals):
            transitions_per_mineral.append(
                prng.choice(possibilities,
                            size=self.minerals_N[i]+20000,
                            p=self.interface_proportions_normalized[i]))

        return tuple(transitions_per_mineral)

    def count_interfaces(self):
        """Count number frequencies of crystal interfaces
        https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/16973510"""

        A = self.interface_pairs

        b = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
        unq_a, unq_cnt = np.unique(b, return_counts=True)
        unq_a = unq_a.view(A.dtype).reshape(-1, A.shape[1])

        return unq_a, unq_cnt

        # Numba does not yet support the return_counts keyword argument of
        # np.unique, unfortunately.

    def convert_counted_interfaces_to_matrix(self):
        """Converts tuple resulting from count_interfaces call to numpy
        matrix. Doesn't break if not all entries of the matrix are present
        in the count.
        """
        count_matrix = np.zeros((len(self.minerals), len(self.minerals)))

        for index, count in zip(*self.interface_counts):
            count_matrix[tuple(index)] = count

        return count_matrix

    def create_interface_location_prob(self):
        """Creates an array descending and then ascending again to
        represent chance of inter crystal breakage of a poly crystalline
        grain (pcg).
        The outer interfaces have a higher chance of breakage than the
        inner ones based on their location within the pcg.
        This represents a linear function.
        Perhaps other functions might be added (spherical) to see the
        effects later on
        """
        size, corr = divmod(len(self.interface_array), 2)
    #     print(size)
    #     print(corr)
        ranger = np.arange(size, 0, -1, dtype=np.uint32)
    #     print(ranger[-2+corr::-1])
        chance = np.append(ranger, ranger[-2+corr::-1])

        return chance

        # Not worth it adding numba to this function

    # To Do: interface strengths matrix still needs to be added here
    # instead of interface_proportions_normalized matrix.
    def get_interface_stregths_prob(self):
        interface_strengths = \
            self.interface_proportions_normalized[self.interface_pairs[:, 0],
                                                  self.interface_pairs[:, 1]]
        return interface_strengths

    def calculate_actual_minerals_N(self):
        minerals_N_total_actual = [np.sum(self.interface_array == i)
                                   for i in range(len(self.minerals))]
        return minerals_N_total_actual

    def create_binned_crystal_size_arrays(self, random_seed=434):
        crystal_size_random = []

        for i, mineral in enumerate(self.minerals):
            crystal_size_random.append(
                np.array(
                    pd.cut(
                        np.exp(
                            self.csds[i].rvs(self.minerals_N_actual[i],
                                             random_state=random_seed)),
                        bins=self.bins,
                        labels=range(len(self.bins)-1),
                        right=False),
                    dtype=np.uint16))

        return crystal_size_random

    def fill_main_cystal_size_array(self, crystal_size_per_mineral):
        """After pre-generation of random crystal_sizes has been
        performed, the sizes are allocated according to the mineral
        order in the minerals/interfaces array
        """
        crystal_size_array = np.zeros(self.interface_array.shape,
                                      dtype=np.uint16)

        # Much faster way (6s) to create crystal size labels array than
        # to use modified function of interfaces array creating (1m40s)!
        for i in range(len(self.minerals)):
            crystal_size_array[np.where(self.interface_array == i)[0]] = \
                crystal_size_per_mineral[i]

        return crystal_size_array

    def get_interface_size_prob(self):
        interface_size_prob = np.min(create_pairs(self.crystal_size_array))
        # Since this represents probabilities, we don't want zeros as a
        # possible value but a 0 size bin exists.
        # Therefore all values are raised by 1 to go around this issue.
        interface_size_prob += 1

        return interface_size_prob


@nb.njit
def calculate_volume_sphere(r, diameter=False):
    """Calculates volume of a sphere
    Numba speeds up this function by 2x
    """
    if diameter:
        r = r/2

    volume = 4/3 * r*r*r * np.pi
    return volume


@nb.njit
def calculate_equivalent_circular_diameter(volume):
    diameter = 2 * (3/4 * volume / np.pi) ** (1/3)

    return diameter


def normalize(data):
    normalized = data / np.sum(data)
    return normalized


def slow_count(data):
    return np.unique(data, return_counts=True)


def fast_count(data):
    return np.bincount(data)


def create_pairs(data):
    return np.dstack((data[:-1],
                      data[1:]))[0]


@nb.jit(nopython=True)
def create_interface_array(minerals_N, transitions_per_mineral):

    # It's faster to work with list here and convert it to array
    # afterwards. 1m13s + 20s compared to 1m45s
    # True, but the list implementation uses around 9 GB memory while
    # the numpy one uses ca. 220 MB as per the array initialization.
    # The time loss of working with a numpy array from the start is
    # around 10 s compared to the list implementation.
    test_array = np.zeros(int(np.sum(minerals_N)), dtype=np.uint8)
#     test_array = [0]
#     append_ = test_array.append
    counters = np.array([0] * len(minerals_N), dtype=np.uint32)
#     array_size_range = range(test_array.shape[0])
    array_size_range = range(int(np.sum(minerals_N)))

    for i in array_size_range:
        previous_state = test_array[i-1]
        test_array[i] = \
            transitions_per_mineral[previous_state][counters[previous_state]]

        counters[previous_state] += 1
#         if i % print_step == 0:
#             print(i, end=" ")

    return test_array

    # Numba provides a ca. 80x times speedup
    # 1.2s compared to 1m45s
