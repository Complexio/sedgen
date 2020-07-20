import numpy as np
import numba as nb
import itertools
from scipy.stats import truncnorm
import time

# Deques support thread-safe, memory efficient appends and pops from
# either side of the deque with approximately the same O(1) performance
# in either direction.
from collections import deque


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
                 csd_means, csd_stds, interfacial_composition=None,
                 learning_rate=1000, timed=False, fast_calc=True, binned=True,
                 volume_binned=False):
        print("---SedGen model initialization started---\n")

        print("Initializing modal mineralogy...")
        self.minerals = minerals
        self.parent_rock_volume = parent_rock_volume
        self.modal_mineralogy = modal_mineralogy
        self.fast_calc = fast_calc
        self.binned = binned
        self.volume_binned = volume_binned

        # Assert that modal mineralogy proportions sum up to unity.
        assert np.isclose(np.sum(modal_mineralogy), 1.0), \
        "Modal mineralogy proportions do not sum to 1"

        self.modal_volume = self.parent_rock_volume * self.modal_mineralogy

        print("Initializing csds...")
        self.csd_means = csd_means
        self.csd_stds = csd_stds

        self.csds = np.array([self.initialize_csd(m)
                              for m in range(len(self.minerals))])
        if self.binned:
            self.size_bins = self.initialize_size_bins()
            self.size_bins_medians = \
                self.calculate_bins_medians(self.size_bins)
            self.n_bins = len(self.size_bins)

            self.volume_bins = self.calculate_volume_bins()
            self.volume_bins_medians = \
                self.calculate_bins_medians(self.volume_bins)

        print("Simulating mineral occurences...", end=" ")
        if timed:
            tic0 = time.perf_counter()
        if not self.fast_calc:
            self.minerals_N, self.simulated_volume, \
            crystal_sizes_per_mineral = \
                self.create_N_crystals(learning_rate=learning_rate)
            # print(self.modal_volume / (self.simulated_volume / 1e9))
        else:
            self.minerals_N, self.simulated_volume = \
                self.create_N_crystals(learning_rate=learning_rate)
        self.N_crystals = np.sum(self.minerals_N)

        if timed:
            toc0 = time.perf_counter()
            print(f" Done in{toc0 - tic0: 1.4f} seconds")
        else:
            print("")

        print("Initializing interfaces...", end=" ")
        if timed:
            tic1 = time.perf_counter()
        self.interfaces = self.get_interface_labels()
        self.interfacial_composition = interfacial_composition

        self.number_proportions = self.calculate_number_proportions()
        self.interface_proportions = self.calculate_interface_proportions()
        self.interface_proportions_normalized = \
            self.calculate_interface_proportions_normalized()
        self.interface_frequencies = self.calculate_interface_frequencies()
        self.interface_frequencies = \
            self.perform_interface_frequencies_correction()

        if not self.fast_calc:
            transitions_per_mineral = \
                self.create_transitions_per_mineral_correctly()
        else:
            transitions_per_mineral = self.create_transitions()

        self.interface_array = \
            create_interface_array(self.minerals_N, transitions_per_mineral)
        self.interface_pairs = create_pairs(self.interface_array)
        if timed:
            toc1 = time.perf_counter()
            print(f" Done in{toc1 - tic1: 1.4f} seconds")
        else:
            print("")

        if timed:
            print("Counting interfaces...", end=" ")
            tic2 = time.perf_counter()
        else:
            print("Counting interfaces...")
        self.interface_counts = self.count_interfaces()
        self.interface_counts_matrix = \
            self.convert_counted_interfaces_to_matrix()
        if timed:
            toc2 = time.perf_counter()
            print(f" Done in{toc2 - tic2: 1.4f} seconds")

        print("Correcting interface arrays for consistency...")
        if not self.fast_calc:
            self.interface_array, self.interface_counts_matrix = \
                self.perform_double_interface_array_correction()
            self.interface_pairs = create_pairs(self.interface_array)

        print("Initializing crystal size array...", end=" ")
        if timed:
            tic3 = time.perf_counter()
        self.minerals_N_actual = self.calculate_actual_minerals_N()
        if self.fast_calc:
            crystal_sizes_per_mineral = \
                self.create_crystal_size_arrays()

        self.crystal_size_array = \
            self.fill_main_cystal_size_array(crystal_sizes_per_mineral)
        if timed:
            toc3 = time.perf_counter()
            print(f" Done in{toc3 - tic3: 1.4f} seconds")
        else:
            print("")

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

    def initialize_csd(self, m, trunc_left=1/256, trunc_right=30):
        mean = np.log(self.csd_means[m])
        std = np.exp(self.csd_stds[m])
        if np.isinf(trunc_left):
            pass
        else:
            trunc_left = np.log(trunc_left)
        trunc_right = np.log(trunc_right)
        a, b = (trunc_left - mean) / std, (trunc_right - mean) / std
        return truncnorm(loc=mean, scale=std, a=a, b=b)

    def initialize_size_bins(self, lower=-10, upper=5, n_bins=1500):
        bins = [2.0**x for x in np.linspace(lower, upper, n_bins+1)]
        return np.array(bins)

    def calculate_bins_medians(self, bins):
        bins_medians = np.array([(bins[i] + bins[i+1]) / 2
                        for i in range(len(bins) - 1)])
        return bins_medians

    # def calculate_bins_medians_volumes(self):
    #     return calculate_volume_sphere(self.bins_medians)

    def calculate_volume_bins(self):
        return calculate_volume_sphere(self.size_bins)

    def calculate_N_crystals(self, m, learning_rate=1000):
        total_volume_mineral = 0
        requested_volume = self.modal_volume[m]
        crystals = deque()
        crystals_append = crystals.append
        crystals_total = deque()
        crystals_total_append = crystals_total.append

        rs = 0
        while total_volume_mineral < requested_volume:
            diff = requested_volume - total_volume_mineral
            crystals_requested = \
                int(diff / (self.modal_mineralogy[m] * learning_rate)) + 1

            crystals_total_append(crystals_requested)
            crystals_to_add = \
                np.exp(self.csds[m].rvs(size=crystals_requested,
                                        random_state=rs))

            if self.volume_binned:
                if not self.fast_calc:
                    crystals_append(calculate_volume_sphere(crystals_to_add))
                total_volume_mineral += \
                    np.sum(crystals[rs])
            else:
                if not self.fast_calc:
                    crystals_append(crystals_to_add)
                total_volume_mineral += \
                    np.sum(calculate_volume_sphere(crystals_to_add))
            rs += 1

        if not self.fast_calc:
            crystals_array = np.concatenate(crystals)

            if self.binned:
                if self.volume_binned:
                    crystals_binned = \
                        (np.digitize(crystals_array,
                                     bins=self.volume_bins) - 1).astype(np.uint16)

                # Capture and correct crystals that fall outside
                # the leftmost bin as they end up as bin 0 but since 1 gets
                # subtracted from all bins they end up as the highest value
                # of np.uint16 as negative values are note possible

                else:
                    crystals_binned = \
                        (np.digitize(crystals_array,
                                     bins=self.size_bins) - 1).astype(np.uint16)

                crystals_binned[crystals_binned > self.n_bins] = 0

                return crystals_total, np.sum(crystals_total), \
                        total_volume_mineral, crystals_binned

            else:
                return crystals_total, np.sum(crystals_total), \
                    total_volume_mineral, crystals_array
        else:
            return crystals_total, np.sum(crystals_total), \
                total_volume_mineral

    def create_N_crystals(self, learning_rate=1000):
        minerals_N = {}
        print("|", end="")
        for m, n in enumerate(self.minerals):
            print(n, end="|")
            minerals_N[n] = \
                self.calculate_N_crystals(m=m, learning_rate=learning_rate)
        minerals_N_total = np.array([N[1] for N in minerals_N.values()])
        simulated_volume = np.array([N[2] for N in minerals_N.values()])
        if not self.fast_calc:
            crystals = np.array([N[3] for N in minerals_N.values()])

            return minerals_N_total, simulated_volume, crystals
        else:
            return minerals_N_total, simulated_volume

    def calculate_number_proportions(self):
        return normalize(self.minerals_N).reshape(-1, 1)

    # To Do: add alpha factor to function to handle non-random interfaces
    def calculate_interface_proportions(self):
        interface_proportions_pred = \
            self.number_proportions * self.number_proportions.T

        return interface_proportions_pred

    def calculate_interface_frequencies(self):
        interface_frequencies = \
            np.round(self.interface_proportions * (self.N_crystals - 1)).astype(np.uint32)
        return interface_frequencies

    def calculate_interface_proportions_normalized(self):
        interface_proportions_normalized = \
            np.divide(self.interface_proportions,
                      np.sum(self.interface_proportions, axis=1).reshape(-1, 1)
                      )
        return interface_proportions_normalized

    def create_transitions(self, random_seed=525):
        possibilities = np.arange(len(self.minerals), dtype=np.uint8)

        prng = np.random.default_rng(random_seed)

        transitions_per_mineral = []

        print("|", end="")
        for i, mineral in enumerate(self.minerals):
            print(mineral, end="|")
            transitions_per_mineral.append(
                prng.choice(possibilities,
                            size=self.minerals_N[i]+20000,
                            p=self.interface_proportions_normalized[i]))

        return tuple(transitions_per_mineral)

    def create_transitions_per_mineral_correctly(self, corr=5,
                                                 random_seed=911):
        """Correction 'corr' is implemented to obtain a bit more possibilities than needed to make sure there are enough values to fill the interface array"""
        transitions_per_mineral = []

        if not self.fast_calc:
            iterable = self.interface_frequencies.copy()
        else:
            iterable = self.interface_proportions_normalized.copy()

        print("|", end="")
        for i, row in enumerate(iterable):
            print(self.minerals[i], end="|")
            # print(self.minerals_N)
            N = self.minerals_N[i].copy() + corr
            prng = np.random.default_rng(random_seed)
            c = prng.random(size=N)
            transitions_per_mineral.append(
                create_transitions_correctly(row, c, N))

        return tuple(transitions_per_mineral)

    def perform_interface_frequencies_correction(self):
        interface_frequencies_corr = self.interface_frequencies.copy()
        diff = np.sum(self.interface_frequencies) - (self.N_crystals - 1)
        # print(diff)
        interface_frequencies_corr[0, 0] -= int(diff)
        return interface_frequencies_corr

    def perform_interface_array_correction(self):
        """Remove or add crystals from/to interface_array where
        necessary
        """
        interface_array_corr = self.interface_array.copy()
        diff = [np.sum(self.interface_array == x) for x in range(6)] - self.minerals_N

        for index, item in enumerate(diff):
            if item > 0:
                interface_array_corr = \
                    np.delete(interface_array_corr,
                              np.where(interface_array_corr == index)
                              [0][-item:])
            elif item < 0:
                interface_array_corr = np.append(interface_array_corr,
                                                 [index] * -item)
            else:
                pass

        return interface_array_corr

    def perform_double_interface_array_correction(self):
        """Remove or add crystals from/to interface_array where
        necessary
        """
        interface_array_corr = self.interface_array.copy()
        prob_unit = 1
        # interface_pairs_corr = self.interface_pairs.copy()
        interface_frequencies_corr = self.interface_counts_matrix.copy()
        diff = [np.sum(self.interface_array == x) for x in range(6)] - self.minerals_N
        # print("diff", diff)
        # print(interface_frequencies_corr)

        for index, item in enumerate(diff):
            if item > 0:
                print("too much", self.minerals[index], item)
                # Select exceeding number crystals from end of array
                for i in range(item):
                    # Select index to correct
                    corr_index = np.where(interface_array_corr == index)[0][-1]

                    # Add/remove interfaces to/from interface_frequencies_corr
                    try:
                        # Remove first old interface
                        pair_index = (interface_array_corr[corr_index-1],
                                      interface_array_corr[corr_index])
                        # print("old1", pair_index)
                        interface_frequencies_corr[pair_index] -= prob_unit
                        # Add newly formed interface
                        pair_index = (interface_array_corr[corr_index-1],
                                      interface_array_corr[corr_index+1])
                        # print("new", pair_index)
                        interface_frequencies_corr[pair_index] += prob_unit
                        # Remove second old interface
                        pair_index = (interface_array_corr[corr_index],
                                      interface_array_corr[corr_index+1])
                        # print("old2", pair_index)
                        interface_frequencies_corr[pair_index] -= prob_unit

                    except IndexError:
                        # print("interface removed from end of array")
                        pass
                        # IndexError might occur if correction takes place at
                        # very end of interface_array, as there is only one
                        # interface present there. Therefore, only the first
                        # old interface needs to be removed. Checking if we are
                        # at the start of the array should not be necessary as # we always select corr_indices from the end of the
                        # interfac_array.

                    # Delete crystals from interface_array_corr
                    interface_array_corr = \
                        np.delete(interface_array_corr, corr_index)

                    # print(interface_array_corr[-100:])
                    # print(interface_frequencies_corr)

            elif item < 0:
                print("too few", self.minerals[index], item)
                # Add newly formed interfaces to interface_frequencies_corr
                pair_index = (interface_array_corr[-1], index)
                interface_frequencies_corr[pair_index] += prob_unit
                # Add crystals to interface_array_corr
                interface_array_corr = np.append(interface_array_corr,
                                                 [index] * -item)
                # Add newly formed isomineral interfaces
                interface_frequencies_corr[index, index] += (-item - 1) * prob_unit
                # print(pair_index)
                # print(interface_array_corr[-100:])
                # print(interface_frequencies_corr)
            else:
                print("all good", self.minerals[index], item)
                pass

        return interface_array_corr, interface_frequencies_corr

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

    def create_crystal_size_arrays(self, random_seed=434):
        crystal_size_random = []

        for i, mineral in enumerate(self.minerals):
            if self.binned:
                if volume_binned:
                    crystals = \
                        (np.digitize(
                            np.exp(
                                self.csds[i].rvs(self.minerals_N_actual[i],
                                                 random_state=random_seed)),
                            bins=self.volume_bins) - 1).astype(np.uint16)
                else:
                    crystals = \
                        (np.digitize(
                            np.exp(
                                self.csds[i].rvs(self.minerals_N_actual[i],
                                                 random_state=random_seed)),
                            bins=self.size_bins) - 1).astype(np.uint16)
            else:
                crystals = np.exp(self.csds[i].rvs(self.minerals_N_actual[i],
                                                   random_state=random_seed))
            crystal_size_random.append(crystals)

        return crystal_size_random

    def fill_main_cystal_size_array(self, crystal_sizes_per_mineral):
        """After pre-generation of random crystal_sizes has been
        performed, the sizes are allocated according to the mineral
        order in the minerals/interfaces array
        """
        if self.binned:
            crystal_size_array = np.zeros(self.interface_array.shape,
                                          dtype=np.uint16)
        else:
            crystal_size_array = np.zeros(self.interface_array.shape,
                                          dtype=np.float64)

        # Much faster way (6s) to create crystal size labels array than
        # to use modified function of interfaces array creating (1m40s)!
        print("|", end="")
        for i, mineral in enumerate(self.minerals):
            print(mineral, end="|")
            crystal_size_array[np.where(self.interface_array == i)] = \
                crystal_sizes_per_mineral[i]

        return crystal_size_array

    def calculate_actual_volumes(self):
        """Calculates the actual volume / modal mineralogy taken up by
        the crystal size array per mineral"""
        actual_volumes = []

        for m in range(len(self.minerals)):
            # Get cystal size (binned) for mineral
            crystal_sizes = self.crystal_size_array[self.interface_array == m]
            # Convert bin labels to bin medians
            if self.binned:
                crystal_sizes_array = self.size_bins_medians[crystal_sizes]
            else:
                crystal_sizes_array = crystal_sizes
            # Calculate sum of volume of crystal sizes and store result
            actual_volumes.append(np.sum(calculate_volume_sphere(crystal_sizes_array)) / self.parent_rock_volume)

        return actual_volumes

    def get_interface_size_prob(self):
        interface_size_prob = np.min(create_pairs(self.crystal_size_array))
        # Since this represents probabilities, we don't want zeros as a
        # possible value but a 0 size bin exists.
        # Therefore all values are raised by 1 to go around this issue.
        interface_size_prob += 1

        return interface_size_prob

    def check_properties(self):
        # Check that number of crystals per mineral in interface array equals
        # the samen number in minerals_N
        assert all([np.sum(self.interface_array == x) for x in range(6)] - \
            self.minerals_N == [0] * len(self.minerals)), "N is not the same in interface_array and minerals_N"
        # Check that
        # assert self.
        return "all good"


@nb.njit(cache=True, nogil=True)
def calculate_volume_sphere(r, diameter=True):
    """Calculates volume of a sphere
    Numba speeds up this function by 2x
    """
    if diameter:
        r = r/2

    volume = 4/3 * r*r*r * np.pi
    return volume


@nb.njit(cache=True, nogil=True)
def calculate_equivalent_circular_diameter(volume):
    diameter = 2 * (3/4 * volume / np.pi) ** (1/3)

    return diameter


def calculate_number_proportions_pcg(pcg_array):
    try:
        pcg_array = np.concatenate(pcg_array)
    except ValueError as error:
        pass
    crystals_count = np.bincount(pcg_array)
    print(crystals_count)
    return crystals_count / np.sum(crystals_count)


def calculate_modal_mineralogy_pcg(pcg_array, csize_array, bins_volumes):
    try:
        pcg_array = np.concatenate(pcg_array)
        csize_array = np.concatenate(csize_array)
    except:
        pass
    volumes = bins_volumes[csize_array]
    volume_counts = weighted_bin_count(pcg_array, volumes)

    return normalize(volume_counts)


@nb.njit(cache=True)
def weighted_bin_count(a, w):
    return np.bincount(a, weights=w)


@nb.njit(cache=True)
def normalize(data):
    return data / np.sum(data)


def slow_count(data):
    return np.unique(data, return_counts=True)


def fast_count(data):
    return np.bincount(data)


def create_pairs(data):
    return np.dstack((data[:-1],
                      data[1:]))[0]


@nb.njit(cache=True)
def create_transitions_correctly(row, c, N_initial):
    """https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s

    Check if probs at end equals all zero to make sure function works.
    """

    # Absolute transition probabilities
    probs = row.copy().astype(np.int32)
    # print(probs)
    # Number of transitions to obtain
    N = int(N_initial)
    # print(N)
    # Subtract correction
    # N_initial -= corr
    # Normalize probabilities
    N_true = np.sum(probs)
    # probs_norm = np.divide(probs, N_true)
    # print(np.sum(probs), N_initial)

    # probs_norm_sum = 1.0
    # print(probs_norm_sum)
    # Initalize transition array
    transitions = np.zeros(shape=N, dtype=np.uint8)

    # For every transition
    for i in range(N):
        # Create normalized probabilities
        # probs_norm = probs / N_initial

        # probs_norm /= probs_norm_sum

        # Create cummulative probability distribution
        # prob_norm_cumsum = np.cumsum(probs_norm)

        # Check were the random probability falls within the cumulative
        # probability distribution and select corresponding mineral
        choice = (c[i] < np.cumsum(probs / N_true)).argmax()

        # Assign corresponding mineral to transition array
        transitions[i] = choice

        # Remove one count of the transition probability array since we want to
        # have the exact number of absolute transitions based on the interface
        # proportions. Similar to a 'replace=False' in random sampling.
        probs[choice] -= 1
        # probs_norm_sum = 1 - (1 / N_initial)
        N_true -= 1
    # print(probs)
    # print(N_true)
    # print(probs_norm_sum)
    return transitions


@nb.njit(cache=True)
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
