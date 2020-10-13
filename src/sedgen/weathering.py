import numpy as np
import numba as nb

from sedgen import sedgen
from sedgen.binning import Bin

import time

"""
TODO:
    - Change intra_cb_p to a function so that smaller crystal sizes have
     a smaller chance of intra_cb breakage and bigger ones a higher
     chance.
    - Implement generalization of intra_cb breakage; instead of
    performing operation per selected mcg in certain bin, perform the
    operation on all selected mcg at same time. This can be done as the
    random location for intra_cb breakage stems from a discrete uniform
    distribution.
    - Would be nice to work with masked arrays for the bin arrays to
    mask negative values, but unfortunately numba does not yet provide
    support for masked arrays.
"""


class Weathering:
    """Prepares an initalized SedGen model for various weathering
    processes and executes them over specified number of timesteps

    Parameters:
    -----------
    model : sedgen.sedgen
        Initalized SedGen model
    n_timesteps : int
        Number of iterations for the for loop which executes the given
        weathering processes
    n_standard_cases : int (optional)
        Number of standard cases to calculate the interface location
        probabilties for; defaults to 2000
    intra_cb_p : list(float) (optional)
        List of probabilities [0, 1] to specify how many of
        mono-crystalline grains per size bin will be effected by
        intra-crystal breakage every timestep; defaults to [0.5] to use
        0.5 for all present mineral classes
    intra_cb_thresholds : list(float) (optional)
        List of intra-crystal breakage size thresholds of mineral
        classes to specify that under the given theshold, intra_crystal
        breakage will not effect the mono-crystalline grains anymore;
        defaults to [1/256] to use 1/256 for all mineral classes
    chem_weath_rates : list(float) (optional)
        List of chemical weathering rates of mineral rates specified as
        'mm/year'. This is scaled internally to be implemented in a
        relative manner; defaults to [0.01] to use 0.01 mm/yr for all
        mineral classes as chemical weathering rate.
    enable_interface_location_prob : bool (optional)
        If True, the location of an interface along a pcg, will have an
        effect on its probability of breakage during inter-crystal
        breakage. Interfaces towards the outside of an pcg are more
        likely to break than those on the inside; defaults to True.
    enable_multi_pcg_breakage : bool (optional)
        If True, during inter-crystal breakage a pcg may break in more
        than two new pcg/mcg grains. This option might speed up the
        model. By activating all interfaces weaker than the selected
        interfaces, this behavior might be accomplished.
    enable_pcg_selection : bool (optional)
        If True, a selection of pcgs is performed to determine which
        pcgs will be affected by inter-crystal breakage during one
        iteration of the weathering procedure. Larger volume pcgs will
        have a higher chance of being selected than smaller ones. If
        enabled, this option probably will slow down the model in
        general.

    """

    def __init__(self, model, n_timesteps, n_standard_cases=2000,
                 intra_cb_p=[0.5], intra_cb_thresholds=[1/256],
                 chem_weath_rates=[0.01], enable_interface_location_prob=True,
                 enable_multi_pcg_breakage=False, enable_pcg_selection=False):
        self.n_timesteps = n_timesteps
        self.n_standard_cases = n_standard_cases
        self.enable_interface_location_prob = enable_interface_location_prob
        self.enable_multi_pcg_breakage = enable_multi_pcg_breakage
        self.enable_pcg_selection = enable_pcg_selection

        self.interface_constant_prob = \
            model.interface_size_prob / model.interface_strengths_prob

        if self.enable_interface_location_prob:
            # Calculate interface_location_prob array for standard
            # configurations of pcgs so that can be looked up later on
            # instead of being calculated ad hoc.
            self.standard_prob_loc_cases = \
                np.array([create_interface_location_prob(
                    np.arange(x)) for x in range(1, n_standard_cases+1)],
                    dtype=np.object)

        self.n_minerals = len(model.minerals)
        self.size_bins = model.size_bins.copy()
        self.size_bins_medians = model.size_bins_medians.copy()
        self.volume_bins = model.volume_bins.copy()
        self.volume_bins_medians = model.volume_bins_medians.copy()
        self.volume_perc_change_unit = \
            self.volume_bins_medians[0] / self.volume_bins_medians[1]
        self.n_bins_medians = self.volume_bins_medians.size
        self.mass_balance_initial = np.sum(model.simulated_volume)
        self.search_size_bins = model.search_size_bins
        self.search_size_bins_medians = model.search_size_bins_medians
        self.ratio_search_size_bins = model.ratio_search_size_bins
        self.search_volume_bins = model.search_volume_bins
        self.search_volume_bins_medians = model.search_volume_bins_medians
        self.ratio_search_volume_bins = model.ratio_search_volume_bins
        # print("mass balance initial:", mass_balance_initial)

        self.pcgs_new = [model.interface_array.copy()]
        self.interface_constant_prob_new = \
            [self.interface_constant_prob.copy()]
        self.crystal_size_array_new = [model.crystal_size_array.copy()]
        self.pcg_chem_weath_array_new = \
            [np.zeros_like(model.interface_array.copy())]
        self.interface_counts = model.interface_counts_matrix.copy()

        self.interface_proportions_normalized = \
            model.interface_proportions_normalized

        self.mcg = \
            np.zeros((self.n_timesteps, self.n_minerals, self.n_bins_medians),
                     dtype=np.uint32)
        # self.mcg_chem_weath = \
        #     np.zeros((self.n_timesteps, self.n_minerals, self.n_bins_medians),
        #              dtype=np.uint32)
        # self.residue_mcg_total = np.zeros(self.n_minerals, dtype=np.float64)
        self.residue = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.float64)
        self.residue_count = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.uint32)

        # Create array of intra-crystal breakage probabilities
        self.intra_cb_p = self.mineral_property_setter(intra_cb_p)

        # Create array of intra-cyrstal breakage size thresholds
        self.intra_cb_thresholds = \
            self.mineral_property_setter(intra_cb_thresholds)

        # Create array of chemical weathering rates
        self.chem_weath_rates = \
            self.mineral_property_setter(chem_weath_rates)

        self.mcg_chem_residue = 0
        self.pcg_chem_residue = 0

        # Model's evolution tracking arrays initialization
        self.pcg_additions = np.zeros(self.n_timesteps, dtype=np.uint32)
        self.mcg_additions = np.zeros(self.n_timesteps, dtype=np.uint32)
        self.mcg_broken_additions = np.zeros(self.n_timesteps, dtype=np.uint32)
        self.residue_additions = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.float64)
        self.residue_count_additions = \
            np.zeros(self.n_timesteps, dtype=np.uint32)
        self.pcg_chem_residue_additions = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.float64)
        self.mcg_chem_residue_additions = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.float64)

        self.pcg_comp_evolution = []
        self.pcg_size_evolution = []

        self.mcg_evolution = \
            np.zeros((self.n_timesteps, self.n_minerals, self.n_bins_medians),
                     dtype=np.uint32)

        # Determine intra-crystal breakage discretization 'rules'
        self.intra_cb_dict, self.intra_cb_breaks, self.diffs_volumes = \
            determine_intra_cb_dict(self.n_bins_medians * 2 - 2,
                                    self.ratio_search_volume_bins)

        self.intra_cb_dict_keys = \
            np.array(list(self.intra_cb_dict.keys()))
        self.intra_cb_dict_values = \
            np.array(list(self.intra_cb_dict.values()))

        # Create bin arrays to capture chemical weathering
        self.size_bins_matrix, self.volume_bins_matrix = \
            self.create_bins_matrix()

        self.size_bins_medians_matrix, self.volume_bins_medians_matrix = \
            self.create_bins_medians_matrix()

        # Volume change array
        self.volume_change_matrix = -np.diff(self.volume_bins_medians_matrix, axis=0)

        # Negative bin array thresholds
        self.negative_volume_thresholds = \
            np.argmax(self.size_bins_medians_matrix > 0, axis=2)

        # Create search_bins_matrix
        self.search_size_bins_matrix, self.search_volume_bins_matrix = \
            self.create_search_bins_matrix()

        # Create search_bins_medians_matrix
        self.search_size_bins_medians_matrix,\
            self.search_volume_bins_medians_matrix = \
            self.create_search_bins_medians_matrix()

        # Create ratio_search_bins_matrix
        self.ratio_search_size_bins_matrix,\
            self.ratio_search_volume_bins_matrix = \
            self.create_ratio_search_bins_matrix()

        # Create array with corresponding bins to intra_cb_thesholds
        # for matrix of bin arrays
        self.intra_cb_threshold_bin_matrix = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.uint16)
        for n in range(self.n_timesteps):
            for m in range(self.n_minerals):
                self.intra_cb_threshold_bin_matrix[n, m] = \
                    np.argmax(self.size_bins_medians_matrix[n, m] >
                              self.intra_cb_thresholds[m])

        # Create intra_cb_dicts for all bin_arrays
        self.intra_cb_breaks_matrix, \
            self.diffs_volumes_matrix = self.create_intra_cb_dicts_matrix()

        self.mass_balance = np.zeros(self.n_timesteps, dtype=np.float64)

    def weathering(self,
                   operations=["intra_cb",
                               "inter_cb",
                               "chem_mcg",
                               "chem_pcg"],
                   display_mass_balance=False,
                   display_mcg_sums=False,
                   timesteps=None):
        if not timesteps:
            timesteps = self.n_timesteps

        mcg_broken = np.zeros_like(self.mcg)
        tac = time.perf_counter()
        # Start model
        for step in range(timesteps):
            # What timestep we're at
            tic = time.perf_counter()
            print(f"{step}/{self.n_timesteps}", end="\r", flush=True)

            # Perform weathering operations
            for operation in operations:
                if operation == "intra_cb":
                    # To Do: Insert check on timestep or n_mcg to
                    # perform intra_cb_breakage per mineral and per bin
                    # or in one operation for all bins and minerals.

                    # intra-crystal breakage
                    mcg_broken, residue, residue_count = \
                        self.intra_crystal_breakage_binned(alternator=step)
                    self.mcg = mcg_broken.copy()
                    # Add new mcg to mcg_chem_weath array to be able to
                    # use newly formed mcg during chemical weathering of
                    # mcg
                    # self.mcg_chem_weath[0] += self.mcg
                    if display_mcg_sums:
                        print("mcg sum over minerals after intra_cb but before inter_cb", np.sum(np.sum(self.mcg, axis=2), axis=0))
                    # Account for residue
                    self.residue[step] = residue
                    self.residue_count[step] = residue_count
                    # print("after intra_cb mcg_vol:", np.sum(self.bins * mcg_broken))
                    # print("after intra_cb residue:", np.sum(residue))
                    toc_intra_cb = time.perf_counter()

                elif operation == "inter_cb":
                    # inter-crystal breakage
                    self.pcgs_new, self.crystal_size_array_new,\
                    self.interface_constant_prob_new, \
                    self.pcg_chem_weath_array_new, self.mcg = \
                        self.inter_crystal_breakage(step)
                    if display_mcg_sums:
                        print("mcg sum after inter_cb", np.sum(np.sum(self.mcg, axis=2), axis=0))
                    toc_inter_cb = time.perf_counter()

                # To Do: Provide option for different speeds of chemical
                # weathering per mineral class. This could be done by
                # moving to a different number of volume bins (n) per
                # mineral class. For the volume_perc_change this would
                # become: volume_perc_change = volume_perc_change ** n
                elif operation == "chem_mcg":
                    # chemical weathering of mcg
                    self.mcg, self.mcg_chem_residue = \
                        self.chemical_weathering_mcg()
                    if display_mcg_sums:
                        print("mcg sum after chem_mcg", np.sum(np.sum(self.mcg, axis=2), axis=0))
                        print("mcg_chem_residue after chem_mcg", self.mcg_chem_residue)
                    toc_chem_mcg = time.perf_counter()

                elif operation == "chem_pcg":
                    # Don't perform chemical weathering of pcg in first
                    # timestep. Otherwise n_timesteps+1 bin arrays need
                    # to be initialized.
                    if step == 0:
                        toc_chem_pcg = time.perf_counter()
                        continue
                    # chemical weathering of pcg
                    self.pcgs_new, \
                        self.crystal_size_array_new, \
                        self.interface_constant_prob_new, \
                        self.pcg_chem_weath_array_new, \
                        self.pcg_chem_residue, \
                        self.interface_counts = \
                        self.chemical_weathering_pcg()
                    toc_chem_pcg = time.perf_counter()

                else:
                    print(f"Warning: {operation} not recognized as a valid operation, skipping {operation} and continueing")
                    continue

            # Track model's evolution
            self.mcg_broken_additions[step] = \
                np.sum([np.sum(x) for x in mcg_broken])  # \
            # - np.sum(self.mcg_broken_additions)
            # self.residue_mcg_total += self.residue
            # print(self.residue[:step])
            # print(self.residue_additions)
            self.residue_additions[step] = self.residue[step]
            # print(self.residue_additions)
            self.residue_count_additions[step] = \
                np.sum(self.residue_count) - \
                np.sum(self.residue_count_additions)

            self.pcg_additions[step] = len(self.pcgs_new)
            self.mcg_additions[step] = np.sum(self.mcg)# - np.sum(mcg_additions)

            self.pcg_comp_evolution.append(self.pcgs_new)
            self.pcg_size_evolution.append(self.crystal_size_array_new)

            self.pcg_chem_residue_additions[step] = self.pcg_chem_residue
            self.mcg_chem_residue_additions[step] = self.mcg_chem_residue

            self.mcg_evolution[step] = np.sum(self.mcg, axis=0)

            # Mass balance check
            if display_mass_balance:
                # mass balance = vol_pcg + vol_mcg + residue
                vol_mcg = np.sum([self.volume_bins_medians_matrix * self.mcg])
                print("vol_mcg_total:", vol_mcg, "over", np.sum(self.mcg), "mcg")
                vol_residue = \
                    np.sum(self.residue_additions) + \
                    np.sum(self.pcg_chem_residue_additions) + \
                    np.sum(self.mcg_chem_residue_additions)
                # print(np.sum(self.residue_additions[step]))
                print("mcg_intra_cb_residue_total:",
                      np.sum(self.residue_additions))
                print("pcg_chem_residue_total:",
                      np.sum(self.pcg_chem_residue_additions))
                print("mcg_chem_residue_total:",
                      np.sum(self.mcg_chem_residue_additions))
                print("vol_residue_total:", vol_residue)

                vol_pcg = self.calculate_vol_pcg()
                # vol_pcg = \
                #     np.sum(
                #         [np.sum(self.volume_bins_medians_matrix[step, m, pcg])
                #         for m, pcg in zip(self.pcgs_new,
                #                           self.crystal_size_array_new)])
                print("vol_pcg_total:", vol_pcg, "over", len(self.pcgs_new), "pcg")

                mass_balance = vol_pcg + vol_mcg + vol_residue
                self.mass_balance[step] = mass_balance
                print(f"new mass balance after step {step}: {mass_balance}\n")

            # If no pcgs are remaining anymore, stop the model
            if not self.pcgs_new:  # Faster to check if pcgs_new has any items
                print(f"After {step} steps all pcg have been broken down to mcg")
                break
            if 'intra_cb' in operations:
                print(f"Intra_cb {step} done in{toc_intra_cb - tic: 1.4f} seconds")
            if 'inter_cb' in operations:
                print(f"Inter_cb {step} done in{toc_inter_cb - toc_intra_cb: 1.4f} seconds")
            if 'chem_mcg' in operations:
                print(f"Chem_mcg {step} done in{toc_chem_mcg - toc_inter_cb: 1.4f} seconds")
            if 'chem_pcg' in operations:
                print(f"Chem_pcg {step} done in{toc_chem_pcg - toc_chem_mcg: 1.4f} seconds")
            print("\n")

            toc = time.perf_counter()
            print(f"Step {step} done in{toc - tic: 1.4f} seconds")
            print(f"Time elapsed: {toc - tac} seconds\n")

        return self.pcgs_new, self.mcg, self.pcg_additions, \
            self.mcg_additions, self.pcg_comp_evolution, \
            self.pcg_size_evolution, self.interface_counts, \
            self.crystal_size_array_new, self.mcg_broken_additions, \
            self.residue_additions, self.residue_count_additions, \
            self.pcg_chem_residue_additions, self.mcg_chem_residue_additions, \
            self.mass_balance, self.mcg_evolution

    def calculate_vol_pcg(self):

        pcg_concat = np.concatenate(self.pcgs_new)
        csize_concat = np.concatenate(self.crystal_size_array_new)
        chem_concat = np.concatenate(self.pcg_chem_weath_array_new)

        vol_pcg = np.sum(self.volume_bins_medians_matrix[chem_concat,
                                                         pcg_concat,
                                                         csize_concat])
        # vol_pcg = 0

        # for m, pcg in zip(self.pcgs_new, self.crystal_size_array_new):
        #     threshold = self.negative_volume_thresholds[step, m]
        #     pcg_filtered = pcg[pcg >= threshold]
        #     vol_pcg += \
        #         np.sum(self.volume_bins_medians_matrix[step, m, pcg_filtered])

        return vol_pcg

    def inter_crystal_breakage(self, step):
        """Performs inter-crystal breakage where poly-crystalline grains
        will break on the boundary between two crystals.

        Parameters:
        -----------
        step : int
            ith iteration of the model (timestep number)

        Returns:
        --------
        pcgs_new : list of np.array(uint8)
            Newly formed list of poly-crystalline grains which are
            represented as seperate numpy arrays
        crystal_size_array_new: list of np.array(uint16)
            Newly formed list of the crystal sizes of the pcgs which are again represented by numpy arrays
        interface_constant_prob_new : list of np.array(float64)
            Newly formed list of the inter-crystal breakage
            probabilities for the present interfaces between crystals in
            seperate pcgs again represented by arrays
        mcg_new : np.array(uint32)
            Newly formed mono-crystalline grains during inter-crystal
            breakage
        """
        pcgs_old = self.pcgs_new
        pcgs_new = []
        pcgs_new_append = pcgs_new.append

        interface_constant_prob_old = self.interface_constant_prob_new
        interface_constant_prob_new = []
        interface_constant_prob_new_append = interface_constant_prob_new.append

        crystal_size_array_old = self.crystal_size_array_new
        crystal_size_array_new = []
        crystal_size_array_new_append = crystal_size_array_new.append

        pcg_chem_weath_array_old = self.pcg_chem_weath_array_new
        pcg_chem_weath_array_new = []
        pcg_chem_weath_array_new_append = pcg_chem_weath_array_new.append

        c_creator = np.random.RandomState(step)
        c = c_creator.random(size=self.pcg_additions[step-1] + 1)

        mcg_temp = [[[]
                    for m in range(self.n_minerals)]
                    for n in range(self.n_timesteps)]
    #         interface_indices = List()

        for i, (pcg, prob, csize, chem) in enumerate(zip(pcgs_old, interface_constant_prob_old, crystal_size_array_old, pcg_chem_weath_array_old)):
            # print(pcg.shape, prob.shape, csize.shape)

            pcg_length = pcg.size

            if self.enable_interface_location_prob:
                # Calculate interface location probability
                if pcg_length <= self.n_standard_cases:
                    location_prob = \
                        self.standard_prob_loc_cases[pcg_length - 1]
                else:
                    location_prob = create_interface_location_prob(pcg)

                # Calculate normalized probability
                probability_normalized = \
                    calculate_normalized_probability(location_prob, prob)

            else:
                probability_normalized = sedgen.normalize(prob)

            # Select interface to break pcg on
            interface = select_interface(i, probability_normalized, c)

            if self.enable_multi_pcg_breakage:
                prob_selected = probability_normalized[interface]
                print(prob_selected)
                interfaces_selected = \
                    np.where(probability_normalized > prob_selected)[0]
                print(interfaces_selected, interfaces_selected.size)

                pcg_new = np.split(pcg, interfaces_selected)
                csize_new = np.split(csize, interfaces_selected)
                chem_new = np.split(chem, interfaces_selected)
                prob_new = np.split(prob, interfaces_selected)

            else:
                # Using indexing instead of np.split is faster.
                # Also avoids the problem of possible 2D arrays instead of
                # 1D being created if array gets split in half.
                # Evuluate first new pcg
                if interface != 1:  # This implies that len(new_prob) != 0
                    pcgs_new_append(pcg[:interface])
                    crystal_size_array_new_append(csize[:interface])
                    pcg_chem_weath_array_new_append(chem[:interface])
                    interface_constant_prob_new_append(prob[:interface-1])
                else:
                    mcg_temp[chem[interface-1]][pcg[interface-1]].append(csize[interface-1])

                # Evaluate second new pcg
                if pcg_length - interface != 1:  # This implies that len(new_prob) != 0
                    pcgs_new_append(pcg[interface:])
                    crystal_size_array_new_append(csize[interface:])
                    pcg_chem_weath_array_new_append(chem[interface:])
                    interface_constant_prob_new_append(prob[interface:])
                else:
                    mcg_temp[chem[interface]][pcg[interface]].append(csize[interface])
                    # print(mcg_temp)

                # Remove interface from interface_counts_matrix
                # Faster to work with matrix than with list and post-loop
                # operations as with the mcg counting
                # print(interface)
                # print(pcg[interface-1], pcg[interface])
                self.interface_counts[pcg[interface-1], pcg[interface]] -= 1
        #             interface_indices.append((pcg[interface-1], pcg[interface]))

        # Add counts from mcg_temp to mcg
        mcg_temp_matrix = np.zeros((self.n_timesteps,
                                    self.n_minerals,
                                    self.n_bins_medians),
                                   dtype=np.uint32)
        for n, outer_list in enumerate(mcg_temp):
            for m, inner_list in enumerate(outer_list):
                # print(type(inner_list), len(inner_list))
                if inner_list:
                    mcg_temp_matrix[n, m] = \
                        np.bincount(inner_list,
                                    minlength=self.n_bins_medians)
                    # sedgen.count_items(inner_list, self.n_bins_medians)
        # mcg_temp_matrix = \
        #     np.asarray([[np.bincount(mcg_temp_list2,
        #                              minlength=self.n_bins_medians)
        #                for mcg_temp_list2 in mcg_temp_list1]
        #                for mcg_temp_list1 in mcg_temp], dtype=np.uint32)

    #         print(mcg_temp_matrix.shape)
    #         for i, mcg_bin_count in enumerate(mcg_bin_counts):
    #             mcg_temp_matrix[i, :len(mcg_bin_count)] = mcg_bin_count
        mcg_new = self.mcg.copy()
        mcg_new += mcg_temp_matrix
        # print([pcg.size for pcg in pcgs_new])
        # print([prob.size for prob in interface_constant_prob_new])
        # print(len(pcgs_new), type(pcgs_new), pcgs_new[0].dtype, pcgs_new[0].shape, np.sum(pcgs_new[0]))
        # print(len(interface_constant_prob_new), type(interface_constant_prob_new), interface_constant_prob_new[0].dtype,
        #     interface_constant_prob_new[0].shape, np.sum(interface_constant_prob_new[0]))

        return pcgs_new, crystal_size_array_new, interface_constant_prob_new, \
            pcg_chem_weath_array_new, mcg_new

    def mineral_property_setter(self, p):
        """Assigns a specified property to multiple mineral classes if
        needed"""

        # TODO:
        # Incorporate option to have a property specified per timestep.

        if len(p) == 1:
            return np.array([p] * self.n_minerals)
        elif len(p) == self.n_minerals:
            return np.array(p)
        else:
            raise ValueError("p should be of length 1 or same length as minerals")

    # TODO: generalize intra_cb_threshold_bin=200 parameter of perform_intra_crystal_breakage_2d; this can be calculated based on intra_cb_threshold and bins.
    def intra_crystal_breakage_binned(self, alternator, start_bin_corr=5):
        mcg_new = np.zeros_like(self.mcg)
        residue_new = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.float64)
        residue_count_new = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype=np.uint32)

        for n in range(self.n_timesteps):
            for m, m_old in enumerate(self.mcg[n]):
                if all(m_old == 0):
                    mcg_new[n, m] = m_old
                else:
                    # print(n, m, np.where(m_old > 0))
                    # m_new, residue_new, residue_count_new = \
                    # perform_intra_crystal_breakage_binned(
                        # m_old, self.intra_cb_p, self.intra_cb_thresholds, i)
                    # print(n, m, m_old.shape)
                    m_new, residue_add, residue_count_add = \
                        perform_intra_crystal_breakage_2d(
                            m_old,
                            self.intra_cb_p,
                            m, n,
                            self.search_volume_bins_medians_matrix[n, m],
                            self.intra_cb_breaks_matrix[n, m],
                            self.diffs_volumes_matrix[n, m],
                            floor=alternator % 2,
                            intra_cb_threshold_bin=self.intra_cb_threshold_bin_matrix[n, m]+start_bin_corr,
                            start_bin_corr=start_bin_corr)
                    mcg_new[n, m] = m_new
                    residue_new[n, m] = residue_add
                    residue_count_new[n, m] = residue_count_add
        # print(residue_new)

        residue_new = np.sum(residue_new, axis=0)
        residue_count_new = np.sum(residue_count_new, axis=0)

        return mcg_new, residue_new, residue_count_new

    def chemical_weathering_mcg(self, shift=1):
        # volume_perc_change = self.volume_perc_change_unit ** shift

        # Reduce size/volume of selected mcg by decreasing their
        # size/volume bin array by one
        mcg_new = np.roll(self.mcg, shift=shift, axis=0)

        # Redisue
        # 1. Residue from mcg being in a negative grain size class
        residue_1 = np.zeros(self.n_minerals, dtype=np.float64)
        for n in range(1, self.n_timesteps):
            for m in range(self.n_minerals):
                threshold = self.negative_volume_thresholds[n, m]
                residue_1[m] += \
                    np.sum(mcg_new[n, m, :threshold] *
                           self.volume_bins_medians_matrix[n-1, m, :threshold])
                # Remove mcg from mcg array that have been added to
                # residue
                mcg_new[n, m, :threshold] = 0

        # 2. Residue from material being dissolved
        # By multiplying the volume change matrix with the already
        # 'rolled' mcg array and summing this over the mineral classes,
        # we end up with the total residue per mineral formed by
        # 'dissolution'.
        residue_2 = \
            np.sum(
                np.sum(mcg_new[1:] * self.volume_change_matrix, axis=0),
                axis=1)
        residue_per_mineral = residue_1 + residue_2

        # residue_per_mineral = 0#\
        #     calculate_mcg_chem_residue(self.mcg,
        #                                self.volume_bins_medians,
        #                                volume_perc_change)

        # # Remove artefact from roll operation
        # mcg_new[:, -shift:] = 0
        # # print(mcg_new)

        return mcg_new, residue_per_mineral

    def chemical_weathering_pcg_old(self, shift=1):

        pcgs_new = []
        pcgs_new_append = pcgs_new.append

        interface_constant_prob_new = []
        interface_constant_prob_new_append = interface_constant_prob_new.append

        crystal_size_array_new = []
        crystal_size_array_new_append = crystal_size_array_new.append

        pcg_chem_weath_array_new = []
        pcg_chem_weath_array_new_append = pcg_chem_weath_array_new.append

        interface_counts_matrix_diff = \
            np.zeros((self.n_minerals, self.n_minerals), dtype=np.int32)
        residue_per_mineral = np.zeros(self.n_minerals, dtype=np.float64)

        for i, (pcg, prob, csize, chem_old) in enumerate(zip(self.pcgs_new, self.interface_constant_prob_new, self.crystal_size_array_new, self.pcg_chem_weath_array_new)):

            # print(pcg.shape, prob.shape, csize.shape, chem.shape)

            # Perform weathering
            chem = chem_old + shift

            # Redisue
            # 1. Residue from mcg being in a negative grain size class
            thresholds = self.negative_volume_thresholds[chem, pcg]
            # print(np.unique(thresholds))
            remaining_crystals = np.where(csize >= thresholds)
            # print(remaining_crystals[0].shape, pcg.shape)

            pcg_remaining = pcg[remaining_crystals]
            csize_remaining = csize[remaining_crystals]
            chem_remaining = chem[remaining_crystals]
            chem_old_remaining = chem_old[remaining_crystals]

            # volumes_old_total += np.sum(volumes_old)
            # volumes = self.volume_bins_medians_matrix[chem, pcg, csize]
            # volumes_total += np.sum(volumes)

            # volumes_pm = \
            #     sedgen.weighted_bin_count(pcg, volumes)
            # volumes_pm = \
            #     np.pad(volumes_pm,
            #            (0, self.n_minerals - len(volumes_pm)))

            # remaining_pm = \
            #     sedgen.weighted_bin_count(pcg_remaining,
            #                               volumes[remaining_crystals])
            # remaining_pm = \
            #     np.pad(remaining_pm,
            #            (0, self.n_minerals - len(remaining_pm)))

            # volumes_per_mineral += volumes_pm
            # remaining_per_mineral += remaining_pm

            # print(pcg_remaining.shape, csize_remaining.shape, chem_remaining.shape)

            # Check if pcg is converted to mcg due to dissolution of
            # crystals and move those pcg to mcg matrix
            if pcg_remaining.size == 1:
                self.mcg[chem_remaining, pcg_remaining, csize_remaining] += 1
            elif pcg_remaining.size == 0:
                # In theory this option should never occur as pcgs are
                # already transfered to mcg when their length equals 1.
                pass
            else:
                # Remove filtered grains
                pcgs_new_append(pcg_remaining)
                crystal_size_array_new_append(csize_remaining)
                pcg_chem_weath_array_new_append(chem_remaining)

                # Several interfaces will be removed and new ones will
                # be added; this thus requires a different strategy!
                # New calculation of interface_strength_prob and
                # interface_size_prob will also be neccesary.
                # remaining_pairs = sedgen.create_pairs(pcg_remaining)

                old_interfaces = \
                    sedgen.count_and_convert_interfaces_to_matrix(
                        pcg, self.n_minerals)
                    # sedgen.convert_counted_interfaces_to_matrix(
                    #     *sedgen.count_interfaces(
                    #         sedgen.create_pairs(pcg)),
                    #     self.n_minerals)

                new_interfaces = \
                    sedgen.count_and_convert_interfaces_to_matrix(
                        pcg_remaining, self.n_minerals)
                    # sedgen.convert_counted_interfaces_to_matrix(
                    #     *sedgen.count_interfaces(remaining_pairs),
                    #     self.n_minerals)

                diff_interfaces = old_interfaces - new_interfaces

                # print(diff_interfaces)

                interface_counts_matrix_diff += diff_interfaces

                interface_size_prob = \
                    sedgen.get_interface_size_prob(csize_remaining)
                interface_strength_prob = \
                    sedgen.get_interface_strengths_prob(
                        self.interface_proportions_normalized,
                        pcg_remaining)

                prob_new = interface_size_prob / interface_strength_prob

                interface_constant_prob_new_append(prob_new)

            # Residue from material being completely dissolved
            dissolved_crystals = np.where(csize < thresholds)
            pcg_dissolved = pcg[dissolved_crystals]
            csize_dissolved = csize[dissolved_crystals]
            chem_old_dissolved = chem_old[dissolved_crystals]
            volumes_old_selected = \
                self.volume_bins_medians_matrix[chem_old_dissolved,
                                                pcg_dissolved,
                                                csize_dissolved]

            residue_1 = \
                sedgen.weighted_bin_count(pcg_dissolved,
                                          volumes_old_selected,
                                          self.n_minerals)
            # residue_1 = \
            #     np.pad(residue_1, (0, self.n_minerals - len(residue_1)))

            # 2. Residue from material being weathered
            dissolved_volume_selected = \
                self.volume_change_matrix[chem_old_remaining,
                                          pcg_remaining,
                                          csize_remaining]
            residue_2 = \
                sedgen.weighted_bin_count(pcg_remaining,
                                          dissolved_volume_selected,
                                          self.n_minerals)
            # residue_2 = \
            #     np.pad(residue_2, (0, self.n_minerals - len(residue_2)))

            # Add residue together per mineral
            residue_per_mineral += residue_1 + residue_2

        # Update interface_counts_matrix
        interface_counts_matrix_new = \
            self.interface_counts + interface_counts_matrix_diff

        return pcgs_new, crystal_size_array_new, interface_constant_prob_new, pcg_chem_weath_array_new, residue_per_mineral, interface_counts_matrix_new

    def chemical_weathering_pcg(self, shift=1):
        """Not taking into account that crystals on the inside of the
        pcg will be less, if even, affected by chemical weathering than
        those on the outside of the pcg"""

        residue_per_mineral = np.zeros(self.n_minerals, dtype=np.float64)
        # remaining_per_mineral = np.zeros(self.n_minerals, dtype=np.float64)
        # volumes_per_mineral = np.zeros(self.n_minerals, dtype=np.float64)
        # volumes_total = 0
        # volumes_old_total = 0

        pcg_lengths = np.array([len(pcg) for pcg in self.pcgs_new],
                               dtype=np.uint32)

        # pcg_lengths = np.zeros(len(self.pcgs_new), dtype=np.uint32)
        # for i, pcg in enumerate(self.pcgs_new):
        #     pcg_lengths[i] = len(pcg)

        pcg_concat = np.concatenate(self.pcgs_new)
        # print(pcg_concat.shape)
        csize_concat = np.concatenate(self.crystal_size_array_new)
        chem_concat_old = np.concatenate(self.pcg_chem_weath_array_new)

        chem_concat = chem_concat_old + 1

        thresholds_concat = self.negative_volume_thresholds[chem_concat, pcg_concat]

        remaining_crystals = csize_concat >= thresholds_concat
        dissolved_crystals = np.where(csize_concat < thresholds_concat)
        # print("n_dissolved_crystals", len(dissolved_crystals[0]))

        pcg_remaining = pcg_concat[remaining_crystals]
        csize_remaining = csize_concat[remaining_crystals]
        chem_remaining = chem_concat[remaining_crystals]
        chem_old_remaining = chem_concat_old[remaining_crystals]

        # pcg_lengths_remaining = np.zeros(len(self.pcgs_new), dtype=np.uint32)

        pcg_filtered = \
            np.split(remaining_crystals, np.cumsum(pcg_lengths[:-1]))
        # count_0 = 0
        # count_1 = 0
        # count_others = 0
        # for i, pcg in enumerate(pcg_filtered):
            # if np.sum(pcg) == 0:
            #     count_0 += 1
            # elif np.sum(pcg) == 1:
            #     count_1 += 1
            # else:
            #     count_others += 1
            # pcg_lengths_remaining[i] = len(pcg[pcg is True])

        pcg_lengths_remaining = \
            np.array([len(pcg[pcg == True]) for pcg in pcg_filtered],
                     dtype=np.uint32)
        # print("count_0", count_0, "count_1", count_1, "count_others", count_others)

        # Need to change length of empty arrays to 1 so that they get
        # deleted properly during the probability operations.
        # pcg_lengths_remaining_for_deletion = pcg_lengths_remaining.copy()
        # pcg_lengths_remaining_for_deletion[pcg_lengths_remaining_for_deletion == 0] = 1

        pcg_lengths_cumul = np.cumsum(pcg_lengths_remaining)
        # pcg_lengths_cumul_for_deletion = pcg_lengths_cumul.copy()
        # pcg_lengths_cumul_for_deletion = \
        #     np.where(pcg_lengths_remaining == 0,
        #              pcg_lengths_cumul + 1,
        #              pcg_lengths_cumul)

        zero_indices = np.where(pcg_lengths_remaining == 0)
        count_0 = zero_indices[0].size
        pcg_lengths_cumul_zero_deleted = np.delete(pcg_lengths_cumul,
                                                   zero_indices)
        # if pcg_lengths_cumul[-1] - pcg_lengths_cumul[-2] <= 1:
        #     pcg_lengths_cumul_for_deletion = \
        #         pcg_lengths_cumul_for_deletion[:-2]
        # else:
        #     pcg_lengths_cumul_for_deletion = \
        #         pcg_lengths_cumul_for_deletion[:-1]

        # pcg_lengths_diff = pcg_lengths - pcg_lengths_remaining

        # print(pcg_lengths, pcg_lengths_diff)
        # print(np.sum(pcg_lengths_diff))
        # print(pcg_remaining[:100])
        # print(np.cumsum(pcg_lengths_remaining[:-1])[:100])
        pcg_remaining_list = \
            np.split(pcg_remaining, pcg_lengths_cumul[:-1])
        csize_remaining_list = \
            np.split(csize_remaining, pcg_lengths_cumul[:-1])
        chem_remaining_list = \
            np.split(chem_remaining, pcg_lengths_cumul[:-1])

        # Mcg accounting
        pcg_to_mcg = \
            pcg_remaining[pcg_lengths_cumul[pcg_lengths_remaining == 1] - 1]
        csize_to_mcg = \
            csize_remaining[pcg_lengths_cumul[pcg_lengths_remaining == 1] - 1]
        chem_to_mcg = \
            chem_remaining[pcg_lengths_cumul[pcg_lengths_remaining == 1] - 1]

        # print(np.sum(self.mcg * self.volume_bins_medians_matrix))
        # print(np.sum(self.mcg))

        # print("mcg_created_volume",
        #       np.sum(self.volume_bins_medians_matrix[chem_to_mcg, pcg_to_mcg, csize_to_mcg]))
        # print(len(pcg_to_mcg), len(csize_to_mcg), len(chem_to_mcg))

        mcg_csize_unq, mcg_csize_ind, mcg_csize_cnt = \
            np.unique(csize_to_mcg, return_index=True, return_counts=True)

        self.mcg[chem_to_mcg[mcg_csize_ind], pcg_to_mcg[mcg_csize_ind], mcg_csize_unq] += mcg_csize_cnt.astype(np.uint32)
        # print(np.sum(self.mcg))
        # if np.sum(self.mcg) == 11605:
        #     raise ValueError
        # print(np.sum(self.mcg * self.volume_bins_medians_matrix))

        # print(len(pcg_remaining_list), len(self.pcgs_new))
        # print(pcg_lengths_remaining)
        # print("new_interface_total", np.sum([length-1 for length in pcg_lengths_remaining[pcg_lengths_remaining > 1]]))
        # print(pcg_to_mcg.shape)

        # Interfaces counts
        # pcg_concat_for_interfaces_prob = pcg_remaining.copy()
        # pcg_concat_for_interfaces[dissolved_crystals] = self.n_minerals
        # print(np.cumsum(pcg_lengths[:-1]).dtype)
        pcg_concat_for_interfaces = \
            np.insert(pcg_remaining,
                      pcg_lengths_cumul[:-1].astype(np.int64),
                      self.n_minerals)
        # print(np.unique(pcg_concat_for_interfaces))
        interface_counts_matrix_new = \
            sedgen.count_and_convert_interfaces_to_matrix(
                pcg_concat_for_interfaces, self.n_minerals)
        # print(interface_counts_matrix_new)
        # print(np.sum(interface_counts_matrix_new))

        # Interface probability calculations
        csize_concat_for_interfaces = \
            csize_remaining.copy().astype(np.int16)
        # csize_concat_for_interfaces[dissolved_crystals] = -1
        # print(csize_remaining.shape)
        csize_concat_for_interfaces = \
            np.insert(csize_concat_for_interfaces,
                      pcg_lengths_cumul[:-1].astype(np.int64),
                      -1)
        # print(csize_concat_for_interfaces.shape)

        interface_size_prob_concat = \
            sedgen.get_interface_size_prob(csize_concat_for_interfaces)
        # print(interface_size_prob_concat.shape)
        interface_size_prob_concat = \
            interface_size_prob_concat[interface_size_prob_concat > 0]
        # print(interface_size_prob_concat.shape)

        interface_strength_prob_concat = \
            sedgen.get_interface_strengths_prob(
                sedgen.expand_array(self.interface_proportions_normalized),
                pcg_concat_for_interfaces)
        # print(interface_strength_prob_concat.shape)
        interface_strength_prob_concat = \
            interface_strength_prob_concat[interface_strength_prob_concat > 0]
        # print(interface_strength_prob_concat.shape)

        prob_remaining = \
            interface_size_prob_concat / interface_strength_prob_concat
        # print(prob_remaining.shape)
        # prob_remaining = np.delete(prob_remaining,
        #                            pcg_lengths_cumul_for_deletion)
        # print(prob_remaining.shape)

        # pcg_lengths_cumul_mod = pcg_lengths_cumul.copy()
        # pcg_lengths_cumul_mod[0] -= 1
        # pcg_lengths_cumul_mod[1:-1] -= np.arange(1, len(self.pcgs_new)-1, dtype=np.uint32
            # )

        prob_remaining_list = \
            np.split(prob_remaining, pcg_lengths_cumul_zero_deleted[:-1]-np.arange(1, len(pcg_remaining_list)-count_0))
        # print(prob_remaining_list[-1].shape)

        # raise Error

        # Residue accounting
        pcg_dissolved = pcg_concat[dissolved_crystals]
        csize_dissolved = csize_concat[dissolved_crystals]
        chem_old_dissolved = chem_concat_old[dissolved_crystals]
        # 1. Residue from mcg being in a negative grain size class
        volumes_old_selected = \
            self.volume_bins_medians_matrix[chem_old_dissolved,
                                            pcg_dissolved,
                                            csize_dissolved]

        residue_1 = \
            sedgen.weighted_bin_count(pcg_dissolved,
                                      volumes_old_selected,
                                      self.n_minerals)
        # residue_1 = \
        #     np.pad(residue_1, (0, self.n_minerals - len(residue_1)))

        # 2. Residue from material being weathered
        dissolved_volume_selected = \
            self.volume_change_matrix[chem_old_remaining,
                                      pcg_remaining,
                                      csize_remaining]
        residue_2 = \
            sedgen.weighted_bin_count(pcg_remaining,
                                      dissolved_volume_selected,
                                      self.n_minerals)
        # residue_2 = \
        #     np.pad(residue_2, (0, self.n_minerals - len(residue_2)))

        # Add residue together per mineral
        residue_per_mineral = residue_1 + residue_2

        # Removing pcg that have been dissolved or have moved to mcg
        # print(pcg_remaining_list[:100])
        pcgs_new = \
            [pcg for pcg in pcg_remaining_list if pcg.size > 1]
        crystal_size_array_new = \
            [pcg for pcg in csize_remaining_list if pcg.size > 1]
        pcg_chem_weath_array_new = \
            [pcg for pcg in chem_remaining_list if pcg.size > 1]
        interface_constant_prob_new = \
            [prob for prob in prob_remaining_list if prob.size != 0]
        # print([pcg.size for pcg in pcgs_new])
        # print([prob.size for prob in interface_constant_prob_new])

        # Check
        # for i, (pcg, prob) in enumerate(zip(pcgs_new, interface_constant_prob_new)):
        #     assert pcg.size == prob.size + 1

        # print(len(pcgs_new), type(pcgs_new), pcgs_new[0].dtype, pcgs_new[0].shape, np.sum(pcgs_new[0]))
        # print(len(interface_constant_prob_new), type(interface_constant_prob_new), interface_constant_prob_new[0].dtype, interface_constant_prob_new[0].shape, np.sum(interface_constant_prob_new[0]))



        # residue_1 = np.zeros(self.n_minerals, dtype=np.float64)
        # for n in range(1, self.n_timesteps):
        #     for m in range(self.n_minerals):
        #         threshold = self.negative_volume_thresholds[n, m]
        #         residue_1[m] += \

        #             np.sum(mcg_new[n, m, :threshold] *
        #                    self.volume_bins_medians_matrix[n-1, m, :threshold])
        #         # Remove mcg from mcg array that have been added to
        #         # residue
        #         # mcg_new[n, m, :threshold] = 0

        #  # 2. Residue from material being dissolved
        # residue = self.volume_change_matrix * self.crystal_size_array_new

        # # chem_pcg_new = np.add(self.pcg_chem_weath_array_new, shift, dtype=np.object)

        # # Need to keep structure of pcgs so can't concatenate here
        # csize_new = \
        #     np.subtract(self.crystal_size_array_new, shift, dtype=np.object)
        # volume_perc_change = self.volume_perc_change_unit ** shift

        # # Also need to keep track of formed residue
        # modal_mineralogy, volumes_old = \
        #     sedgen.calculate_modal_mineralogy_pcg(self.pcgs_new,
        #                                           self.crystal_size_array_new,
        #                                           self.volume_bins_medians)

        # old_volume = np.sum(volumes_old)
        # residue = old_volume * (1 - volume_perc_change)

        # residue_per_mineral = residue * modal_mineralogy
        # print("volumes_old", volumes_old_total)
        # print("volumes", volumes_total)
        # print("volumes_r", np.sum(volumes_per_mineral))
        # print("remaining", np.sum(remaining_per_mineral))

        return pcgs_new, crystal_size_array_new, interface_constant_prob_new, pcg_chem_weath_array_new, residue_per_mineral, interface_counts_matrix_new

    def create_bins_matrix(self):
        """Create the matrix holding the arrays with bins which each
        represent the inital bin array minus x times the chemical
        weathering rate per mineral class.
        """

        size_bins_matrix = \
            np.array([[self.size_bins - x * self.chem_weath_rates[i]
                      for i in range(self.n_minerals)]
                     for x in range(self.n_timesteps)]
                     )
        # if masked:
        #     mask = size_bins_matrix < 0
        #     size_bins_matrix = ma.masked_array(size_bins_matrix, mask)

        volume_bins_matrix = sedgen.calculate_volume_sphere(size_bins_matrix)

        return size_bins_matrix, volume_bins_matrix

    def create_bins_medians_matrix(self):
        size_bins_medians_matrix = \
            sedgen.calculate_bins_medians(self.size_bins_matrix)
        volume_bins_medians_matrix = \
            sedgen.calculate_bins_medians(self.volume_bins_matrix)

        return size_bins_medians_matrix, volume_bins_medians_matrix

    def create_search_bins_matrix(self):
        search_size_bins_matrix = \
            np.array([[self.search_size_bins - x * self.chem_weath_rates[i]
                      for i in range(self.n_minerals)]
                     for x in range(self.n_timesteps)]
                     )
        # if masked:
        #     mask = search_size_bins_matrix < 0
        #     search_size_bins_matrix = \
        #         ma.masked_array(search_size_bins_matrix, mask)

        search_volume_bins_matrix = \
            sedgen.calculate_volume_sphere(search_size_bins_matrix)

        return search_size_bins_matrix, search_volume_bins_matrix

    def create_search_bins_medians_matrix(self):
        search_size_bins_medians_matrix = \
            sedgen.calculate_bins_medians(self.search_size_bins_matrix)

        search_volume_bins_medians_matrix = \
            sedgen.calculate_bins_medians(self.search_volume_bins_matrix)

        return search_size_bins_medians_matrix, search_volume_bins_medians_matrix

    def create_ratio_search_bins_matrix(self):
        ratio_search_size_bins_matrix = \
            calculate_ratio_search_bins_matrix(
                self.search_size_bins_medians_matrix)
        ratio_search_volume_bins_matrix = \
            calculate_ratio_search_bins_matrix(
                self.search_volume_bins_medians_matrix)

        return ratio_search_size_bins_matrix, ratio_search_volume_bins_matrix

    def create_intra_cb_dicts_matrix(self):
        # Need to account for 'destruction' of geometric series due to
        # chemical weathering --> implement chemical weathering rates
        # into the function somehow.
        # intra_cb_dicts_matrix = \
        #     np.array([[[determine_intra_cb_dict(b, self.ratio_search_volume_bins_matrix[n, m])
        #              for b in range(self.n_bins_medians+1000,
        #                             self.n_bins_medians*2)]
        #              for m in range(self.n_minerals)]
        #              for n in range(self.n_timesteps)],
        #              dtype='object')

        # intra_cb_dicts_matrix = \
        #     np.zeros((self.n_timesteps, self.n_minerals), dtype='object')
        intra_cb_breaks_matrix = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype='object')
        diffs_volumes_matrix = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype='object')

        for n in range(self.n_timesteps):
            print(n)
            for m in range(self.n_minerals):
                # print("\t", m)

                # intra_cb_dict_array = \
                #     np.zeros(self.n_bins_medians - self.intra_cb_threshold_bin_matrix[n, m],
                #             dtype='object')
                intra_cb_breaks_array = \
                    np.zeros((self.n_bins_medians - self.intra_cb_threshold_bin_matrix[n, m],
                        len(self.intra_cb_breaks)),
                            dtype=np.uint16)
                diffs_volumes_array = \
                    np.zeros((self.n_bins_medians - self.intra_cb_threshold_bin_matrix[n, m],
                        len(self.intra_cb_breaks)),
                            dtype=np.float64)

                for i, b in \
                    enumerate(range(self.intra_cb_threshold_bin_matrix[n, m] +\
                                    self.n_bins_medians,
                                    self.n_bins_medians*2)):
                    intra_cb_breaks_array[i], diffs_volumes_array[i] = \
                        determine_intra_cb_dict_array_version(b, self.ratio_search_volume_bins_matrix[n, m],
                            max_n_values=len(self.intra_cb_breaks))
                # intra_cb_dicts_matrix[n, m] = intra_cb_dict_array
                intra_cb_breaks_matrix[n, m] = intra_cb_breaks_array
                diffs_volumes_matrix[n, m] = diffs_volumes_array

        # intra_cb_breaks_matrix = intra_cb_dicts_matrix[..., 1]
        # diffs_volumes_matrix = \
        #     intra_cb_dicts_matrix[..., 2]
        # intra_cb_dicts_matrix = intra_cb_dicts_matrix

        return intra_cb_breaks_matrix, diffs_volumes_matrix

    def calculate_mass_balance_difference(self):
        return self.mass_balance[1:] - self.mass_balance[:-1]


def calculate_ratio_search_bins_matrix(search_bins_medians_matrix):
    return search_bins_medians_matrix / \
        search_bins_medians_matrix[..., -1, None]


def create_interface_location_prob(a):
    """Creates an array descending and then ascending again to represent
    chance of inter crystal breakage of a poly crystalline grain (pcg).
    The outer interfaces have a higher chance of breakage than the
    inner ones based on their location within the pcg.
    This represents a linear function.
    Perhaps other functions might be added (spherical) to see the
    effects later on.
    """
    size, corr = divmod(a.size, 2)
#     print(size)
#     print(corr)
    ranger = np.arange(size, 0, -1, dtype=np.uint32)
#     print(ranger[-2+corr::-1])
    chance = np.append(ranger, ranger[-2+corr::-1])

    return chance

# Not worth it adding numba to this function


# Speedup from 6m30s to 2m45s
# Not parallelizable
@nb.njit(cache=True)
def select_interface(i, probs, c):
    interface = (c[i] < np.cumsum(probs)).argmax() + 1
    # The '+ 1' makes sure that the first interface can also be selected
    # Since the interface is used to slice the interface_array,
    # interface '0' would result in the pcg not to be broken at all
    # since e.g.:
    # np.array([0, 1, 2, 3, 4])[:0] = np.array([])

    return interface


# Speedup from 2m45s to 1m30s
# Parallelizable but not performant for small pcg
@nb.njit(cache=True)
def calculate_normalized_probability(location_prob, prob):
    probability = location_prob * prob
    return probability / np.sum(probability)


def perform_intra_crystal_breakage_2d(mcg_old, prob, mineral_nr, timestep,
                                      search_bins,
                                      intra_cb_breaks, diffs_volumes,
                                      intra_cb_threshold_bin=200, floor=True,
                                      verbose=False, start_bin_corr=5,
                                      intra_cb_dict_keys=None,
                                      intra_cb_dict_values=None):
    # Certain percentage of mcg has to be selected for intra_cb
    # Since mcg are already binned it doesn't matter which mcg get
    # selected in a certain bin, only how many

    mcg_new = mcg_old.copy()
    n_bins = mcg_new.size
#     print("pre-intra_cb volume:", np.sum(search_bins[-1500:] * mcg_new))
    residue_count = 0
    residue_new = 0

    # 1. Select mcg
    if floor:
        # 1st time selection
        mcg_selected = np.floor(mcg_new * prob[mineral_nr]).astype(np.uint32)
    else:
        # 2nd time selection
        mcg_selected = np.ceil(mcg_new * prob[mineral_nr]).astype(np.uint32)
    # print(mcg_selected[mcg_selected > 0])
    # print(np.sum(mcg_selected))

    # Sliced so that only the mcg above the intra_cb_threshold_bin are
    # affected; same reasoning in for loop below.
    mcg_new[intra_cb_threshold_bin:] -= mcg_selected[intra_cb_threshold_bin:]

    # n_max = np.max(mcg_selected[intra_cb_threshold_bin:])
    # breaker = \
    #     np.random.randint(low=1,
    #                       high=len(intra_cb_breaks),
    #                       size=n_max)\
    #     .astype(np.uint16)

    # rand_int = np.random.default_rng()

    # 2. Create break points
    for i, n in enumerate(mcg_selected[intra_cb_threshold_bin:]):
        if n == 0:
            continue
        intra_cb_breaks_to_use = \
            intra_cb_breaks[i+start_bin_corr][diffs_volumes[i+start_bin_corr] > 0]
        diffs_volumes_to_use = \
            diffs_volumes[i+start_bin_corr][diffs_volumes[i+start_bin_corr] > 0]

        breaker_size, breaker_remainder = \
            divmod(n, intra_cb_breaks_to_use.size)

        breaker_counts = np.array([breaker_size] * intra_cb_breaks_to_use.size,
                                  dtype=np.uint32)
        breaker_counts[-1] += breaker_remainder

        # if intra_cb_breaks_to_use.size == 1:
        #     breaker = np.zeros(n, dtype=np.uint16)
        # else:
        # # Why does 'low' have to be equal to 1??
        #     breaker = rand_int.integers(low=0,
        #                                 high=intra_cb_breaks_to_use.size,
        #                                 size=n,
        #                                 dtype=np.uint8)
        #         # np.random.randint(low=0,
        #         #                   high=intra_cb_breaks_to_use.size,
        #         #                   size=n,
        #         #                   dtype=np.uint16)
        # breaker_counts = sedgen.bin_count(breaker).astype(np.uint32)
        # print("breaker_counts.shape", breaker_counts.shape)
        p1 = i + intra_cb_threshold_bin \
            - np.arange(1, breaker_counts.size+1)
        p2 = p1 - intra_cb_breaks_to_use
        # print(p1.shape)
        # print(p2.shape)

        mcg_new[p1] += breaker_counts
        mcg_new[p2] += breaker_counts

        # print(intra_cb_breaks_to_use)
        # print(diffs_volumes_to_use)
        # print(breaker_counts)
        residue_new += \
            np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes_to_use * breaker_counts)

        # print(np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes_to_use[:breaker_counts.size] * breaker_counts))
        # print(np.sum(diffs_volumes_to_use[:breaker_counts.size] * breaker_counts))

        # residue_new += np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes_to_use[breaker])

        # print(np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes_to_use[breaker]))
        # print(np.sum(diffs_volumes_to_use[breaker]))

        # if i == 5:
        #     break

        # p1 = i + intra_cb_threshold_bin + n_bins - breaker - 1
        # print(np.unique(p1))
        # print(breaker, n, intra_cb_breaks_to_use[breaker])
        # p2 = p1 - intra_cb_breaks_to_use[breaker]
        # if verbose and len(p1) != 0:
            # print(i, intra_cb_threshold_bin, n_bins, breaker)
            # print(intra_cb_dict_keys[breaker[:n]]-n_bins+intra_cb_threshold_bin+2+i)
            # print(p1)
            # print(intra_cb_dict_values[breaker[:n]]-n_bins+intra_cb_threshold_bin+2+i)
            # print(p2)
        # p = np.concatenate((p1, p2))

        # p_filtered = p[np.where(p >= n_bins)]
        # print(p_filtered.size, p.size)
        # if p_filtered.size != 0:
        # p_count = np.bincount(p - n_bins).astype(np.uint32)
        # print(p)
        # print(intra_cb_breaks, intra_cb_breaks_to_use)
        # p_count = \
        #     sedgen.bin_count(p - n_bins).astype(np.uint32)
        # print(p_count[750:])
            # p_count = sedgen.count_items(p_filtered - n_bins, n_bins).astype(np.uint32)
        # mcg_new[:p_count.size] += p_count
#             if i in [500, 1000]:
#                 print(n * search_bins[i + intra_cb_threshold_bin + n_bins])
#                 print(np.sum(p_count * search_bins[-1500:len(p_count) + 1500]) + np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes[breaker]))
#         else:
#             if n == 0:
#                 print(i, "no mcg of this size to break")
#             else:
#                 print(i, "all mcg to residue")

        # Account for formed residue
        # If difference between initial bin and smallest formed bin during
        # intra_cb is lower than intra_cb_threshold_bin, no mcg can be formed
        # that would direcly fall in the residue bins.
        # Since this will never be the case for our use cases, we can
        # disable this check.
        # if intra_cb_breaks_to_use[0] > intra_cb_threshold_bin:
        #     residue = p[np.where(p < n_bins)]
        #     residue_count += residue.size
        #     residue_new += np.sum(search_bins[residue])

        # Addition of small fraction of material that
        # gets 'lost' during intra_cb_breakage to residue.
        # if verbose and len(p1) != 0:
        #     print("ok", search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes_to_use[breaker])
        # residue_new += np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes_to_use[breaker])
    # print("residue_intra_cb", residue_new, "for", np.sum(mcg_selected), "selected crystal(s) for mineral", mineral_nr, "in timestep", timestep)

#     print("post-intra_cb volume:", np.sum(search_bins[-1500:] * mcg_new) + residue_new)
#     print("residue_new:", residue_new)
    return mcg_new, residue_new, residue_count


@nb.njit(cache=True)
def calculate_mcg_chem_residue(mcg_old, bins, volume_perc_change):
    """Calculates the residue produced during chemical weathering coming
    from mono-crystalline grains.

    Parameters:
    -----------
    mcg_old : np.array
        Old mcg array/matrix
    bins : np.array
        Bin values to use
    volume_perc_change : float
        Volume percentage change, determined by 'shift' and
        "volume_perc_change_unit"

    Returns:
    --------
    residue_per_mineral : np.array
        Amount of residue formed per mineral class
    """
    # Keep track of residue formed by shifting bins
    residue_per_mineral = \
        np.sum(mcg_old[:, 1:] * bins[1:], axis=1) * (1 - volume_perc_change)
    # Add total volume of first bin to residue
    residue_per_mineral += mcg_old[:, 0] * bins[0]

    return residue_per_mineral


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
        sum of part 1 and part 2 of broken mcg. This thus represents the percentage of the formed intra_cb residue with regard to the
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

    # half_bin_number = int(len(specific_ratios) * 0.5)
    # print(half_bin_number)
    # print(specific_ratios[:half_bin_number])

    # numba not possible for this option since axis kwarg is not
    # supported for np.argmax
    # bin1 = np.arange(bin_label-1, 0, -1)
    # bin2_ratio = (1 - specific_ratios[bin1]).reshape(-1, 1)
    # bin2 = np.argmax(bin2_ratio < specific_ratios, axis=1) - corr
    # filter_ = np.where((bin1 - bin2 >= 0) & (bin2 != -1))
    # bin1_f, bin2_f = \
    #     bin1[filter_].astype(np.uint16), bin2[filter_].astype(np.uint16)

    # intra_cb_dict = dict(zip(bin1_f, bin2_f))
    # actual_n_values = len(intra_cb_dict)

    # if not max_n_values:
    #     max_n_values = actual_n_values

    # diffs = np.zeros(max_n_values, dtype=np.uint16)
    # diffs_volumes = np.zeros(max_n_values, dtype=np.float64)

    # diffs[:actual_n_values] = bin1_f - bin2_f
    # diffs_volumes[:actual_n_values] = \
    #     1 - (specific_ratios[bin1_f] + specific_ratios[bin2_f])

    for i, bin1 in enumerate(range(bin_label-1, 0, -1)):
        bin2_ratio = 1 - specific_ratios[bin1]
        # Minus 1 for found bin so that volume sum of two new mcg
        # is bit less than 100%; remainder goes to residue later on.
        # bin2 = np.argmax(bin2_ratio < specific_ratios) - corr
        bin2 = find_closest(bin2_ratio, specific_ratios, corr=corr)
        if bin2 == -1:
            break
        # if verbose:
        #     print(bin1, specific_ratios[bin1], bin2_ratio, bin2, bin1 - bin2)
        #     print(specific_ratios[bin1] + specific_ratios[bin2])
        intra_cb_dict[bin1] = bin2
        if max_n_values:
            diffs[i] = bin1 - bin2
            diffs_volumes[i] = \
                bin2_ratio - specific_ratios[bin2]
                # 1 - (specific_ratios[bin1] + specific_ratios[bin2])
        else:
            diffs.append(bin1 - bin2)
            diffs_volumes.append(bin2_ratio - specific_ratios[bin2])
            # diffs_volumes.append(1 - (specific_ratios[bin1] + specific_ratios[bin2]))
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
        # bin2 = np.argmax(bin2_ratio < specific_ratios) - corr
        bin2 = find_closest(bin2_ratio, specific_ratios, corr=corr)
        if bin2 == -1:
            break

        diffs[i] = bin1 - bin2
        diffs_volumes[i] = bin2_ratio - specific_ratios[bin2]
        # 1 - (specific_ratios[bin1] + specific_ratios[bin2])
        if (diffs[i] <= corr):
            break

    return diffs, diffs_volumes

# def determine_intra_cb_dict_list_version():


@nb.njit(cache=True)
def find_closest(value, lookup, corr=0):
    return np.argmax(value < lookup) - corr


### OLD CODE ###
@nb.njit
def account_interfaces(interface_counts, interface_indices):
    for interface_index in interface_indices:
        interface_counts[interface_index] -= 1
    return interface_counts


@nb.njit
def perform_intra_crystal_breakage(i, m_old, p):
    # Create binary array to select crystals that will be broken
    selection = np.random.binomial(1, p[i], m_old.size) == 1
    n_not_selected = np.sum((selection == 0))
    #         print(len(m_old) + np.sum(selection))

    m_new = np.zeros(shape=len(m_old) + np.sum(selection), dtype=np.float64)
    #         print(m_new.shape)
    #         print(np.sum(selection))
    #         print(np.sum(selection == False))

    # Add non-selected crystals to m_new array
    m_new[:n_not_selected] = m_old[~selection]

    # Get diameters of selected crystals
    selected_diameters = m_old[selection]

    # Calculate volume of crystals that will break
    selected_volumes = sedgen.calculate_volume_sphere(selected_diameters, diameter=True)

    # Create [0, 1] array to determine where crystals will break
    cutter = np.random.random(selected_volumes.size)

    # Break crystals by dividing the volume according to [0, 1] array
    broken_volumes_1 = selected_volumes * cutter
    broken_volumes_2 = selected_volumes * (1 - cutter)

    broken_volumes = np.concatenate((broken_volumes_1, broken_volumes_2))

    # Calculate corresponding crystal diameter (and size bin)
    broken_diameters = sedgen.calculate_equivalent_circular_diameter(broken_volumes)

    # Add newly formed mcg to m_new array
    m_new[n_not_selected:] = broken_diameters
        # If newly formed crystal is smaller than smallest size bin
        # volume should be added to residue array
    # Threshold value based on clay particle diameter (8 on phi scale)
    condition = m_new > 0.00390625
    m_new_filtered = m_new[condition]
    residue_new = np.sum(sedgen.calculate_volume_sphere(m_new[~condition], diameter=True))
    residue_count = m_new[~condition].size
    return m_new_filtered, residue_new, residue_count


def intra_crystal_breakage(mcg_old, p, residue, residue_count):
    # Initialize new mcg array and residue array
    mcg_new = []

    for i, m_old in enumerate(mcg_old):
        if len(m_old) == 0:
            mcg_new.append([])
        else:
            m_new, residue_new, residue_count_new = perform_intra_crystal_breakage.py_func(i, m_old, p)
            mcg_new.append(list(m_new))
            residue[i] += residue_new
            residue_count[i] += residue_count_new
    return mcg_new, residue, residue_count


# Approach for one mineral class m of mcg
# bins_medians_volumes = sedgen_CA_NS.volume_bins_medians.copy()
# bins = sedgen_CA_NS.volume_bins.copy()

@nb.njit(cache=True)
def perform_intra_crystal_breakage_binned(tally, p, intra_cb_thresholds, i, residue_threshold=1/1024):
#     rng = np.random.default_rng(seed=random_seed)
#     print(tally.dtype)

    # Filter bins below intra-crystal breakage threshold
    binlabel_at_threshold = (bins > sedgen.calculate_volume_sphere(intra_cb_thresholds[i])).argmax() - 1
    tally_above_threshold = tally[binlabel_at_threshold:]
    tally_below_threshold = np.zeros_like(tally, dtype=np.uint32)
    tally_below_threshold[:binlabel_at_threshold] = tally[:binlabel_at_threshold]
#     print(np.sum(tally_below_threshold[:binlabel_at_threshold]))

    n_mcg = np.sum(tally_above_threshold)

    # Order selection according to positioning in m array
    indices = np.arange(binlabel_at_threshold, bins_medians_volumes.size)
#     print(indices)
    tally_flat = np.repeat(indices, tally_above_threshold)
    # Get binomial selection
#     selection = np.random.binomial(1, p, n_mcg) == 1
#     random_floats = rng.random(n_mcg)
    random_floats = np.random.random(n_mcg)
    selection = random_floats < p[i]

    n_selected = np.sum(selection)
    n_not_selected = n_mcg - n_selected

    mcg_new = np.zeros(shape=n_mcg+n_selected, dtype=np.uint16)
    # Perform selection of m array based on ordered selection array
    # Move operation downstream to save memory allocation
#     selected_items = tally_flat[selection]
    # Non selected diamaters should remain in new mcg array
    mcg_new[:n_not_selected] = tally_flat[~selection]
    # Get cutter according
    cutter = random_floats[:n_selected]

    # Get selected volumes based on bins_medians_volumes
    selected_volumes = bins_medians_volumes[tally_flat[selection]]
    # Multiply cutter with correct index (= volume class) of binned_medians_volumes
    # Multiply (1 -cutter) with correct index (= volume class) of binned_medians_volumes
    broken_volumes_1 = selected_volumes * cutter
    broken_volumes_2 = selected_volumes - broken_volumes_1
    broken_volumes = np.concatenate((broken_volumes_1, broken_volumes_2))

    # Check for values that should be moved to residue array based on
    # threshold value
    residue_condition = broken_volumes > sedgen.calculate_volume_sphere(residue_threshold)

    broken_volumes_filtered = broken_volumes[residue_condition]
    n_selected_filtered = len(broken_volumes_filtered)
    residue = broken_volumes[~residue_condition]
    residue_count = len(residue)
    residue = np.sum(residue)

    # Calculate diamaters of filtered broken volumes
    # This step is not needed as size_bins and volume_bins are the same except
    # for the fact that they represent the proporty of size in a different
    # dimension
#     broken_diameters = sedgen.calculate_equivalent_circular_diameter(broken_volumes_filtered)
    # Bin broken diameters according to bins of model
#     broken_diameters_binned = np.digitize(broken_diameters, bins=bins) - 1
#     broken_diameters_binned = np.digitize(broken_volumes_filtered, bins=bins) - 1
    broken_diameters_binned = np.searchsorted(bins, broken_volumes_filtered) - 1

    # Correct for values that fall outside of leftmost bin
    # This should not be a problem here, however, as during the residue check
    # all values smaller than the leftmost bin have been moved from the broken
    # array to the residue array
    broken_diameters_binned[broken_diameters_binned > len(bins)] = 0

    # Add filtered broken diamaters to mcg_new array
    mcg_new[n_not_selected:n_not_selected+n_selected_filtered] = broken_diameters_binned

    # Remove non-used items from array
    mcg_new = np.delete(mcg_new, slice(n_not_selected+n_selected_filtered, len(mcg_new)))

    # Obtain new tally array
    tally_new = np.zeros(bins_medians_volumes.size, dtype=np.uint32)
    tally_count = np.bincount(mcg_new)
    tally_new[:len(tally_count)] = tally_count

    # Add unaffected tally below intra_cb_threshold to new tally
    tally_new += tally_below_threshold

    return tally_new, residue, residue_count


# Approach for one mineral class m of mcg
# bins_medians_volumes = sedgen_CA_NS.volume_bins_medians.copy()
# bins = sedgen_CA_NS.volume_bins.copy()

# n_search_bins = 1500
# search_bins = sedgen.calculate_volume_sphere(np.array([2.0**x for x in np.linspace(-25, 5, n_bins*2+1)]))
# search_bins_medians = np.array([(search_bins[i] + search_bins[i+1]) / 2
#                                 for i in range(n_bins * 2 - 1)])
# ratio_search_bins = search_bins_medians / search_bins_medians[-1]

@nb.njit(cache=True)
def perform_intra_crystal_breakage_binned_v2(tally, p, intra_cb_thresholds, i, residue_threshold=1/1024):


#     rng = np.random.default_rng(seed=random_seed)
#     print(tally.dtype)

    # Filter bins below intra-crystal breakage threshold
    binlabel_at_threshold = (bins > sedgen.calculate_volume_sphere(intra_cb_thresholds[i])).argmax() - 1
    tally_above_threshold = tally[binlabel_at_threshold:]
    tally_below_threshold = np.zeros_like(tally, dtype=np.uint32)
    tally_below_threshold[:binlabel_at_threshold] = tally[:binlabel_at_threshold]
#     print(np.sum(tally_below_threshold[:binlabel_at_threshold]))

    n_mcg = np.sum(tally_above_threshold)

    # Order selection according to positioning in m array
    indices = np.arange(binlabel_at_threshold, bins_medians_volumes.size)
#     print(indices)
    tally_flat = np.repeat(indices, tally_above_threshold)
    # Get binomial selection
#     selection = np.random.binomial(1, p, n_mcg) == 1
#     random_floats = rng.random(n_mcg)
    random_floats = np.random.random(n_mcg)
    selection = random_floats < p[i]

    n_selected = np.sum(selection)
    n_not_selected = n_mcg - n_selected

    mcg_new = np.zeros(shape=n_mcg+n_selected, dtype=np.uint16)
    # Perform selection of m array based on ordered selection array
    # Move operation downstream to save memory allocation
    selected_items = tally_flat[selection]
    # Non selected diamaters should remain in new mcg array
    mcg_new[:n_not_selected] = tally_flat[~selection]
    # Get cutter according
    cutter = random_floats[:n_selected]

    # Get selected volumes based on bins_medians_volumes
#     selected_volumes = bins_medians_volumes[tally_flat[selection]]
    # Multiply cutter with correct index (= volume class) of binned_medians_volumes
    # Multiply (1 -cutter) with correct index (= volume class) of binned_medians_volumes
#     broken_volumes_1 = selected_volumes * cutter
#     broken_volumes_2 = selected_volumes - broken_volumes_1
#     broken_volumes = np.concatenate((broken_volumes_1, broken_volumes_2))
    print(cutter.shape)
    p1 = np.searchsorted(ratio_search_bins, cutter) - (1500 - selected_items)
    p2 = np.searchsorted(ratio_search_bins, 1 - cutter) - (1500 - selected_items) - 1
    broken_volumes = np.concatenate((p1, p2))

    # Check for values that should be moved to residue array based on
    # bin threshold value
    residue_condition = broken_volumes > n_search_bins

    # Check for values that should be moved to residue array based on
    # threshold value
#     residue_condition = broken_volumes > sedgen.calculate_volume_sphere(residue_threshold)

    broken_volumes_filtered = broken_volumes[residue_condition]
    n_selected_filtered = len(broken_volumes_filtered)
    residue = broken_volumes[~residue_condition]
    residue_count = len(residue)
    residue = np.sum(search_bins_medians[residue])

    # Calculate diamaters of filtered broken volumes
    # This step is not needed as size_bins and volume_bins are the same except
    # for the fact that they represent the proporty of size in a different
    # dimension
#     broken_diameters = sedgen.calculate_equivalent_circular_diameter(broken_volumes_filtered)
    # Bin broken diameters according to bins of model
#     broken_diameters_binned = np.digitize(broken_diameters, bins=bins) - 1
#     broken_diameters_binned = np.digitize(broken_volumes_filtered, bins=bins) - 1
#     broken_diameters_binned = np.searchsorted(bins, broken_volumes_filtered) - 1

    # Correct for values that fall outside of leftmost bin
    # This should not be a problem here, however, as during the residue check
    # all values smaller than the leftmost bin have been moved from the broken
    # array to the residue array
#     broken_diameters_binned[broken_diameters_binned > len(bins)] = 0

    # Add filtered broken diamaters to mcg_new array
    mcg_new[n_not_selected:n_not_selected+n_selected_filtered] = broken_volumes_filtered - n_search_bins

    # Remove non-used items from array
    mcg_new = np.delete(mcg_new, slice(n_not_selected+n_selected_filtered, len(mcg_new)))

    # Obtain new tally array
    tally_new = np.zeros(bins_medians_volumes.size, dtype=np.uint32)
    tally_count = np.bincount(mcg_new)
    tally_new[:len(tally_count)] = tally_count

    # Add unaffected tally below intra_cb_threshold to new tally
    tally_new += tally_below_threshold

    return tally_new, residue, residue_count


def count_size_evolution(pcg_comp_evolution, pcg_size_evolution, n_minerals, n_bins):
    size_count_matrix = np.zeros((len(pcg_comp_evolution), n_minerals, n_bins), dtype=np.uint32)

    for m in range(n_minerals):
        size_count = np.unique(pcg_size_evolution[0][0][np.where(pcg_comp_evolution[0][0] == 0)], return_counts=True)
