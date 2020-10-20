import numpy as np
import numba as nb
import itertools
from scipy.stats import truncnorm
import time

from sedgen import general as gen


"""TODO:
    - Provide option to specify generated minerals based on a number
      instead of filling up a given volume. In the former case, the
      simulated volume attribute also has more meaning.
    - Learning rate should be set semi-automatically. Based on smallest
      mean crystal size perhaps?
"""


class SedGen:
    """Initializes a SedGen model based on fundamental properties of
    modal mineralogy, interfacial composition and crystal size
    statistics

    Parameters:
    -----------
    minerals : list
        Mineral classes to use in model; the order in which the minerals
        are specified here should be kept accros all other parameters
    parent_rock_volume : float
        Volume representing parent rock to fill with crystals at start
        of model
    modal_mineralogy : np.array
        Volumetric proportions of mineral classes at start of model
    csd_means : np.array
        Crystal size means of mineral classes
    csd_stds : np.array
        Crystal size standard deviations of mineral classes
    interfacial_composition : np.array (optional)
    learning_rate : int (optional)
        Amount of change used during determination of N crystals per
        mineral class; defaults to 1000
    timed : bool (optional)
        Show timings of various initialization steps; defaults to False
    """

    def __init__(self, minerals, parent_rock_volume, modal_mineralogy,
                 csd_means, csd_stds, interfacial_composition=None,
                 learning_rate=1000, timed=False):
        print("---SedGen model initialization started---\n")

        print("Initializing modal mineralogy...")
        self.minerals = minerals
        self.n_minerals = len(self.minerals)
        self.parent_rock_volume = parent_rock_volume
        self.modal_mineralogy = modal_mineralogy

        # Assert that modal mineralogy proportions sum up to unity.
        assert np.isclose(np.sum(modal_mineralogy), 1.0), \
            "Modal mineralogy proportions do not sum to 1"

        # Divide parent rock volume over all mineral classes based on
        # modal mineralogy
        self.modal_volume = self.parent_rock_volume * self.modal_mineralogy

        print("Initializing csds...")
        self.csd_means = csd_means
        self.csd_stds = csd_stds

        self.csds = np.array([self.initialize_csd(m)
                              for m in range(self.n_minerals)])

        self.size_bins = \
            gen.initialize_size_bins()
        self.size_bins_medians = \
            gen.calculate_bins_medians(self.size_bins)

        self.n_bins = len(self.size_bins)
        self.n_bins_medians = self.n_bins - 1

        self.volume_bins = \
            gen.calculate_volume_sphere(self.size_bins)
        self.volume_bins_medians = \
            gen.calculate_bins_medians(self.volume_bins)

        self.search_size_bins = \
            gen.initialize_search_size_bins(self.n_bins)
        self.search_size_bins_medians = \
            gen.calculate_bins_medians(self.search_size_bins)
        self.ratio_search_size_bins = \
            gen.calculate_ratio_search_bins(self.search_size_bins_medians)

        self.search_volume_bins = \
            gen.calculate_volume_sphere(self.search_size_bins)
        self.search_volume_bins_medians = \
            gen.calculate_bins_medians(self.search_volume_bins)
        self.ratio_search_volume_bins = \
            gen.calculate_ratio_search_bins(
                self.search_volume_bins_medians)

        print("Simulating mineral occurences...", end=" ")
        if timed:
            tic0 = time.perf_counter()
        self.minerals_N, self.simulated_volume, crystal_sizes_per_mineral = \
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

        transitions_per_mineral = \
            self.create_transitions_per_mineral_correctly()

        self.interface_array = \
            create_interface_array(self.minerals_N, transitions_per_mineral)
        # self.interface_pairs = gen.create_pairs(self.interface_array)
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

        self.interface_counts_matrix = \
            gen.count_and_convert_interfaces_to_matrix(self.interface_array,
                                                       self.n_minerals)
        if timed:
            toc2 = time.perf_counter()
            print(f" Done in{toc2 - tic2: 1.4f} seconds")

        print("Correcting interface arrays for consistency...")
        self.interface_array, self.interface_counts_matrix = \
            self.perform_interface_array_correction()

        print("Initializing crystal size array...", end=" ")
        if timed:
            tic3 = time.perf_counter()
        self.minerals_N_actual = self.calculate_actual_minerals_N()

        self.crystal_size_array = \
            self.fill_main_cystal_size_array(crystal_sizes_per_mineral)
        if timed:
            toc3 = time.perf_counter()
            print(f" Done in{toc3 - tic3: 1.4f} seconds")
        else:
            print("")

        print("Initializing inter-crystal breakage probability arrays...")
        # ???
        # Probability arrays need to be normalized so that they carry
        # the same weight (importance) down the line. During
        # calculations with the probility arrays, weights might be added
        # as coefficients to change the importance of the different
        # arrays as required.
        # ???

        # The more an interface is located towards the outside of a
        # grain, the more chance it has to be broken.
        self.interface_location_prob = self.create_interface_location_prob()
        # The higher the strength of an interface, the less chance it
        # has to be broken.
        self.interface_strengths_prob = \
            gen.get_interface_strengths_prob(
                self.interface_proportions_normalized,
                self.interface_array)
        # The bigger an interface is, the more chance it has to be
        # broken.
        self.interface_size_prob = \
            gen.get_interface_size_prob(self.crystal_size_array)

        print("\n---SedGen model initialization finished succesfully---")

    def __repr__(self):
        output = f"SedGen({self.minerals}, {self.parent_rock_volume}," \
                 f"{self.modal_mineralogy}, {self.csd_means}," \
                 f"{self.csd_stds}, {self.interfacial_composition}," \
                 f"{self.learning_rate}"
        return output

    def get_interface_labels(self):
        """Returns list of combinations of interfaces between provided
        list of minerals

        """

        interface_labels = \
            ["".join(pair) for pair in
             itertools.combinations_with_replacement(self.minerals, 2)]

        return interface_labels

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

        mean = np.log(self.csd_means[m])
        std = np.exp(self.csd_stds[m])

        if not np.isinf(trunc_left):
            trunc_left = np.log(trunc_left)

        if not np.isinf(trunc_right):
            trunc_right = np.log(trunc_right)

        a, b = (trunc_left - mean) / std, (trunc_right - mean) / std
        csd = truncnorm(loc=mean, scale=std, a=a, b=b)

        return csd

    def calculate_N_crystals(self, m, learning_rate=1000):
        """Request crystals from CSD until accounted modal volume is
        filled.

        Idea: use pdf to speed up process --> This can only be done if
        the CSD is converted to a 'crystal volume distribution'.

        From this, the number of crystals per mineral class will be
        known while also giving the total number of crystals (N) in 1 m³
        of parent rock.
        """
        total_volume_mineral = 0
        requested_volume = self.modal_volume[m]
        crystals = []
        crystals_append = crystals.append
        crystals_total = 0
        # crystals_total_append = crystals_total.append

        rs = 0
        while total_volume_mineral < requested_volume:
            diff = requested_volume - total_volume_mineral
            crystals_requested = \
                int(diff / (self.modal_mineralogy[m] * learning_rate)) + 1

            crystals_total += crystals_requested
            crystals_to_add = \
                np.exp(self.csds[m].rvs(size=crystals_requested,
                                        random_state=rs))

            crystals_append(gen.calculate_volume_sphere(crystals_to_add))
            total_volume_mineral += np.sum(crystals[rs])

            rs += 1

        crystals_array = np.concatenate(crystals)

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

    def create_N_crystals(self, learning_rate=1000):
        minerals_N = {}
        print("|", end="")
        for m, n in enumerate(self.minerals):
            print(n, end="|")
            minerals_N[n] = \
                self.calculate_N_crystals(m=m, learning_rate=learning_rate)
        minerals_N_total = np.array([N[1] for N in minerals_N.values()])
        simulated_volume = np.array([N[2] for N in minerals_N.values()])
        crystals = [N[3] for N in minerals_N.values()]

        return minerals_N_total, simulated_volume, crystals

    def calculate_number_proportions(self):
        """Returns number proportions"""
        return gen.normalize(self.minerals_N).reshape(-1, 1)

    # To Do: add alpha factor to function to handle non-random interfaces
    def calculate_interface_proportions(self):
        interface_proportions_pred = \
            self.number_proportions * self.number_proportions.T

        return interface_proportions_pred

    def calculate_interface_frequencies(self):
        interface_frequencies = \
            np.round(self.interface_proportions * (self.N_crystals - 1))\
              .astype(np.uint32)
        return interface_frequencies

    def calculate_interface_proportions_normalized(self):
        interface_proportions_normalized = \
            np.divide(self.interface_proportions,
                      np.sum(self.interface_proportions, axis=1).reshape(-1, 1)
                      )
        return interface_proportions_normalized

    def create_transitions_per_mineral_correctly(self, corr=5,
                                                 random_seed=911):
        """Correction 'corr' is implemented to obtain a bit more
        possibilities than needed to make sure there are enough values
        to fill the interface array later on."""
        transitions_per_mineral = []

        iterable = self.interface_frequencies.copy()

        prng = np.random.default_rng(random_seed)
        print("|", end="")
        for i, row in enumerate(iterable):
            print(self.minerals[i], end="|")
            N = self.minerals_N[i] + corr
            c = prng.random(size=N)
            transitions_per_mineral.append(
                create_transitions_correctly(row, c, N))

        return tuple(transitions_per_mineral)

    def perform_interface_frequencies_correction(self):
        interface_frequencies_corr = self.interface_frequencies.copy()
        diff = np.sum(self.interface_frequencies) - (self.N_crystals - 1)
        interface_frequencies_corr[0, 0] -= int(diff)

        return interface_frequencies_corr

    def perform_interface_array_correction(self):
        """Remove or add crystals from/to interface_array where
        necessary
        """
        interface_array_corr = self.interface_array.copy()
        prob_unit = 1
        # interface_pairs_corr = self.interface_pairs.copy()
        interface_frequencies_corr = self.interface_counts_matrix.copy()
        diff = [np.sum(self.interface_array == x) for x in range(6)] \
            - self.minerals_N
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
                        # at the start of the array should not be necessary as
                        # we always select corr_indices from the end of the
                        # interface_array.

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
                print("all good", self.minerals[index], item)
                pass

        return interface_array_corr, interface_frequencies_corr

    def create_interface_location_prob(self):
        """Creates an array descending and then ascending again to
        represent chance of inter crystal breakage of a poly crystalline
        grain (pcg).
        The outer interfaces have a higher chance of breakage than the
        inner ones based on their location within the pcg.
        This represents a linear function.
        Perhaps other functions might be added (spherical) to see the
        effects later on

        # Not worth it adding numba to this function
        """
        size, corr = divmod(self.interface_array.size, 2)
        ranger = np.arange(size, 0, -1, dtype=np.uint32)
        chance = np.append(ranger, ranger[-2+corr::-1])

        return chance

    def calculate_actual_minerals_N(self):
        minerals_N_total_actual = [np.sum(self.interface_array == i)
                                   for i in range(self.n_minerals)]
        return minerals_N_total_actual

    def fill_main_cystal_size_array(self, crystal_sizes_per_mineral):
        """After pre-generation of random crystal_sizes has been
        performed, the sizes are allocated according to the mineral
        order in the minerals/interfaces array
        """
        crystal_size_array = np.zeros(self.interface_array.shape,
                                      dtype=np.uint16)

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

        for m in range(self.n_minerals):
            # Get cystal size (binned) for mineral
            crystal_sizes = self.crystal_size_array[self.interface_array == m]
            # Convert bin labels to bin medians
            crystal_sizes_array = self.size_bins_medians[crystal_sizes]
            # Calculate sum of volume of crystal sizes and store result
            actual_volumes.append(
                np.sum(
                    gen.calculate_volume_sphere(
                        crystal_sizes_array)) / self.parent_rock_volume)

        return actual_volumes

    def check_properties(self):
        # Check that number of crystals per mineral in interface dstack
        # array equals the same number in minerals_N
        assert all([np.sum(self.interface_array == x) for x in range(6)]
                   - self.minerals_N == [0] * self.n_minerals), \
                   "N is not the same in interface_array and minerals_N"

        return "Number of crystals (N) is the same in interface_array and"
        "minerals_N"


def calculate_number_proportions_pcg(pcg_array):
    """Calculates the number proportions of the mineral classes present

    Parameters:
    -----------
    pcg_array : np.array
        Array holding the mineral identities of the crystals part of
        poly-crystalline grains

    Returns:
    --------
    number_proportions : np.array
        Normalized number proportions of crystals forming part of
        poly-crystalline grains
    """
    try:
        pcg_array = np.concatenate(pcg_array)
    except ValueError:
        pass
    crystals_count = np.bincount(pcg_array)
    print(crystals_count)
    number_proportions = gen.normalize(crystals_count)
    return number_proportions


def calculate_modal_mineralogy_pcg(pcg_array, csize_array, bins_volumes,
                                   return_volumes=True):
    """Calculates the volumetric proportions of the mineral classes
    present.

    Parameters:
    -----------
    pcg_array : np.array
        Array holding the mineral identities of the crystals part of
        poly-crystalline grains
    csize_array : np.array
        Array holding the crystal sizes in bin labels
    bins_volumes : np.array
        Bins to use for calculation of the crystal's volumes
    return_volumes : bool (optional)
        Whether to return the calculated volumes of the crystal or not;
        defaults to True

    Returns:
    --------
    modal_mineralogy: np.array
        Volumetric proportions of crystals forming part of
        poly-crystalline grains
    volumes : np.array
        Volumes of crystals forming part of poly-crystalline grains
    """
    try:
        pcg_array = np.concatenate(pcg_array)
        csize_array = np.concatenate(csize_array)
    except ValueError:
        pass

    volumes = bins_volumes[csize_array]
    volume_counts = gen.weighted_bin_count(pcg_array, volumes)
    modal_mineralogy = gen.normalize(volume_counts)

    if return_volumes:
        return modal_mineralogy, volumes
    else:
        return modal_mineralogy


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
    test_array = np.zeros(int(np.sum(minerals_N)), dtype=np.uint8)
    counters = np.array([0] * len(minerals_N), dtype=np.uint32)
    array_size_range = range(int(np.sum(minerals_N)))

    for i in array_size_range:
        previous_state = test_array[i-1]
        test_array[i] = \
            transitions_per_mineral[previous_state][counters[previous_state]]

        counters[previous_state] += 1

    return test_array

    # Numba provides a ca. 80x times speedup
    # 1.2s compared to 1m45s