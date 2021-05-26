import numpy as np
import numba as nb
from scipy.stats import truncnorm

import itertools

from sedgen import general as gen


class MineralOccurenceMixin:
    def __init__(self):
        self.pr_minerals_N, self.pr_simulated_volume, \
            self.pr_crystal_sizes_per_mineral = \
            self.create_N_crystals()

        self.pr_mass_balance_initial = np.sum(self.pr_simulated_volume)
        self.pr_N_crystals = np.sum(self.pr_minerals_N)

    def create_N_crystals(self):
        minerals_N = {}
        print("|", end="")
        for m, n in enumerate(self.pr_minerals):
            print(n, end="|")
            minerals_N[n] = \
                self.simulate_N_crystals(m=m)
        minerals_N_total = np.array([N[1] for N in minerals_N.values()])
        simulated_volume = np.array([N[2] for N in minerals_N.values()])
        crystals = [N[3] for N in minerals_N.values()]

        return minerals_N_total, simulated_volume, crystals

    def simulate_N_crystals(self, m):
        """Request crystals from CSD until accounted modal volume is
        filled.

        Idea: use pdf to speed up process --> This can only be done if
        the CSD is converted to a 'crystal volume distribution'.

        From this, the number of crystals per mineral class will be
        known while also giving the total number of crystals (N) in 1 m³
        of parent rock.
        """
        total_volume_mineral = 0
        requested_volume = self.pr_modal_volume[m]
        crystals = []
        crystals_append = crystals.append
        crystals_total = 0
        # crystals_total_append = crystals_total.append

        rs = 0
        while total_volume_mineral < requested_volume:
            # volume that still needs to be filled with crystals
            diff = requested_volume - total_volume_mineral
            # ‘m’ represents the current mineral class
            # +1 so that each time at least one crystal is requested
            crystals_requested = \
                int(diff / (self.pr_modal_mineralogy[m] * self.learning_rate)) + 1
            # print(crystals_requested, end=", ")

            crystals_total += crystals_requested
            crystals_to_add = \
                np.exp(self.pr_csds[m].rvs(size=crystals_requested,
                                           random_state=rs))

            crystals_append(gen.calculate_volume_sphere(crystals_to_add))
            total_volume_mineral += np.sum(crystals[rs])
            # print(requested_volume, total_volume_mineral)

            rs += 1

        try:
            crystals_array = np.concatenate(crystals)
        except ValueError:
            crystals_array = np.array(crystals)

        crystals_binned = \
            (np.searchsorted(self.volume_bins,
                             crystals_array) - 1).astype(np.uint16)

        # Capture and correct crystals that fall outside
        # the leftmost bin as they end up as bin 0 but since 1 gets
        # subtracted from all bins they end up as the highest value
        # of np.uint16 as negative values are not possible
        crystals_binned[crystals_binned > self.n_bins] = 0

        return crystals_total, np.sum(crystals_total), \
            total_volume_mineral, crystals_binned


class InterfaceOccurenceMixin:
    def __init__(self):
        self.pr_interfaces = self.get_interface_labels()
        self.pr_number_proportions = self.calculate_number_proportions()
        self.pr_interface_proportions = self.calculate_interface_proportions()
        self.pr_interface_proportions_normalized = \
            self.calculate_interface_proportions_normalized()
        self.pr_interface_frequencies = self.calculate_interface_frequencies()
        self.pr_interface_frequencies = \
            self.perform_interface_frequencies_correction()

        self.pr_transitions_per_mineral = \
            self.create_transitions_per_mineral_correctly()

        self.pr_crystals = \
            create_interface_array(self.pr_minerals_N,
                                   self.pr_transitions_per_mineral)

    def get_interface_labels(self):
        """Returns list of combinations of interfaces between provided
        list of minerals

        """

        interface_labels = \
            ["".join(pair) for pair in
             itertools.combinations_with_replacement(self.pr_minerals, 2)]

        return interface_labels

    def calculate_number_proportions(self):
        """Returns number proportions"""
        return gen.normalize(self.pr_minerals_N).reshape(-1, 1)

    # To Do: add alpha factor to function to handle non-random interfaces
    def calculate_interface_proportions(self):
        if self.pr_interfacial_composition:
            interface_proportions_true = self.pr_interfacial_composition
            return interface_proportions_true

        else:
            interface_proportions_pred = \
                self.pr_number_proportions * self.pr_number_proportions.T
            return interface_proportions_pred

    def calculate_interface_frequencies(self):
        interface_frequencies = \
            np.round(self.pr_interface_proportions * (self.pr_N_crystals - 1))\
              .astype(np.uint32)
        return interface_frequencies

    def calculate_interface_proportions_normalized(self):
        interface_proportions_normalized = \
            np.divide(self.pr_interface_proportions,
                      np.sum(self.pr_interface_proportions,
                             axis=1).reshape(-1, 1)
                      )
        return interface_proportions_normalized

    def create_transitions_per_mineral_correctly(self, corr=100,
                                                 random_seed=911):
        """Correction 'corr' is implemented to obtain a bit more
        possibilities than needed to make sure there are enough values
        to fill the interface array later on."""
        transitions_per_mineral = []

        iterable = self.pr_interface_frequencies.copy()

        if self.fixed_random_seeds:
            prng = np.random.default_rng(random_seed)
        else:
            prng = np.random.default_rng()
        print("|", end="")
        for i, row in enumerate(iterable):
            print(self.pr_minerals[i], end="|")
            N = self.pr_minerals_N[i] + corr
            c = prng.random(size=N)
            transitions_per_mineral.append(
                create_transitions_correctly(row, c, N))

        return tuple(transitions_per_mineral)

    def perform_interface_frequencies_correction(self):
        interface_frequencies_corr = self.pr_interface_frequencies.copy()
        diff = np.sum(self.pr_interface_frequencies) - (self.pr_N_crystals - 1)
        interface_frequencies_corr[0, 0] -= int(diff)

        return interface_frequencies_corr

    def perform_interface_array_correction(self):
        """Remove or add crystals from/to interface_array where
        necessary
        """
        interface_array_corr = self.pr_crystals.copy()
        prob_unit = 1
        # interface_pairs_corr = self.interface_pairs.copy()
        interface_frequencies_corr = self.pr_interface_counts_matrix.copy()
        diff = [np.sum(self.pr_crystals == x)
                for x in range(self.pr_n_minerals)] - self.pr_minerals_N
        # print("diff", diff)
        # print(interface_frequencies_corr)

        for index, item in enumerate(diff):
            if item > 0:
                print("too much", self.pr_minerals[index], item)
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
                        # at the start of the array should not be necessary as
                        # we always select corr_indices from the end of the
                        # interface_array.

                    # Delete crystals from interface_array_corr
                    interface_array_corr = \
                        np.delete(interface_array_corr, corr_index)

                    # print(interface_array_corr[-100:])
                    # print(interface_frequencies_corr)

            elif item < 0:
                print("too few", self.pr_minerals[index], item)
                # Add newly formed interfaces to interface_frequencies_corr
                pair_index = (interface_array_corr[-1], index)
                interface_frequencies_corr[pair_index] += prob_unit
                # Add crystals to interface_array_corr
                interface_array_corr = \
                    np.concatenate((interface_array_corr,
                                    np.array([index] * -item, dtype=np.uint8)))
                # Add newly formed isomineral interfaces
                interface_frequencies_corr[index, index] += \
                    (-item - 1) * prob_unit
                # print(pair_index)
                # print(interface_array_corr[-100:])
                # print(interface_frequencies_corr)
            else:
                print("all good", self.pr_minerals[index], item)
                pass

        return interface_array_corr, interface_frequencies_corr


@nb.njit(cache=True)
def create_transitions_correctly(row, c, N_initial):
    """https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s

    Check if freqs at end equals all zero to make sure function works.
    """

    # Absolute transition probabilities
    freqs = row.copy().astype(np.int32)

    # Number of transitions to obtain
    N = int(N_initial)

    # Normalize probabilities
    N_true = np.sum(freqs)

    # Initalize transition array
    transitions = np.zeros(shape=N, dtype=np.uint8)

    # For every transition
    for i in range(N):
        # Check were the random probability falls within the cumulative
        # probability distribution and select corresponding mineral
        choice = (c[i] < np.cumsum(freqs / N_true)).argmax()

        # Assign corresponding mineral to transition array
        transitions[i] = choice

        # Remove one count of the transition frequency array since we want to
        # have the exact number of absolute transitions based on the interface
        # proportions. Similar to a 'replace=False' in random sampling.
        freqs[choice] -= 1
        N_true -= 1

    return transitions


@nb.njit(cache=True)
def create_interface_array(minerals_N, transitions_per_mineral):
    # It's faster to work with list here and convert it to array
    # afterwards. 1m13s + 20s compared to 1m45s
    # True, but the list implementation uses around 9 GB memory while
    # the numpy one uses ca. 220 MB as per the array initialization.
    # The time loss of working with a numpy array from the start is
    # around 10 s compared to the list implementation.
    interface_array = np.zeros(int(np.sum(minerals_N)), dtype=np.uint8)
    counters = np.zeros(len(minerals_N), dtype=np.uint32)
    array_size_range = range(int(np.sum(minerals_N)))

    for i in array_size_range:
        previous_state = interface_array[i-1]
        interface_array[i] = \
            transitions_per_mineral[previous_state][counters[previous_state]]
        if interface_array[i] > 5:
            print(i, previous_state, counters[previous_state], transitions_per_mineral[previous_state][counters[previous_state]], transitions_per_mineral[previous_state])
        counters[previous_state] += 1

    return interface_array

    # Numba provides a ca. 80x times speedup
    # 1.2s compared to 1m45s


class CrystalSizeMixin:
    def __init__(self):
        self.pr_csds = np.array([self.initialize_csd(m)
                                for m in range(self.pr_n_minerals)])

    def initialize_csd(self, m, trunc_left=1/256, trunc_right=30):
        """Initalizes the truncated lognormal crystal size distribution

        Parameters:
        -----------
        m : int
            Number specifying mineral class
        trunc_left : float(optional)
            Value to truncate lognormal distribution to on left side,
            i.e. smallest values
        trunc_right : float(optional)
            Value to truncate lognormal distribution to on right side,
            i.e. biggest values

        Returns:
        --------
        csd : scipy.stats.truncnorm
            Truncated lognormal crystal size distribution
        """

        mean = np.log(self.pr_csd_means[m])
        std = np.exp(self.pr_csd_stds[m])

        if not np.isinf(trunc_left):
            trunc_left = np.log(trunc_left)

        if not np.isinf(trunc_right):
            trunc_right = np.log(trunc_right)

        a, b = (trunc_left - mean) / std, (trunc_right - mean) / std
        csd = truncnorm(loc=mean, scale=std, a=a, b=b)

        return csd

    def fill_main_cystal_size_array(self):
        """After pre-generation of random crystal_sizes has been
        performed, the sizes are allocated according to the mineral
        order in the minerals/interfaces array
        """
        crystal_size_array = np.zeros(self.pr_crystals.shape,
                                      dtype=np.uint16)

        # Much faster way (6s) to create crystal size labels array than
        # to use modified function of interfaces array creating (1m40s)!
        print("|", end="")
        for i, mineral in enumerate(self.pr_minerals):
            print(mineral, end="|")
            crystal_size_array[np.where(self.pr_crystals == i)] = \
                self.pr_crystal_sizes_per_mineral[i]

        return crystal_size_array
