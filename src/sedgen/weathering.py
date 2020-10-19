import numpy as np
import numba as nb
import time

from sedgen import general as gen


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
            # configurations of pcgs so that they can be looked up later on
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

        # --------------------------------------------------------------
        # MOVE TO INITIALIZATION

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
        self.volume_change_matrix = -np.diff(self.volume_bins_medians_matrix,
                                             axis=0)

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
        # --------------------------------------------------------------

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
                    # TODO: Insert check on timestep or n_mcg to
                    # perform intra_cb_breakage per mineral and per bin
                    # or in one operation for all bins and minerals.

                    # intra-crystal breakage
                    mcg_broken, residue, residue_count = \
                        self.intra_crystal_breakage_binned(alternator=step)
                    self.mcg = mcg_broken.copy()
                    # Add new mcg to mcg_chem_weath array to be able to
                    # use newly formed mcg during chemical weathering of
                    # mcg
                    if display_mcg_sums:
                        print("mcg sum over minerals after intra_cb but before inter_cb", np.sum(np.sum(self.mcg, axis=2), axis=0))
                    # Account for residue
                    self.residue[step] = residue
                    self.residue_count[step] = residue_count
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

                    # If no pcgs are remaining anymore, stop the model
                    if not self.pcgs_new:
                        print(f"After {step} steps all pcg have been broken down to mcg")
                        return self.pcgs_new, self.mcg, self.pcg_additions, \
                            self.mcg_additions, self.pcg_comp_evolution, \
                            self.pcg_size_evolution, self.interface_counts, \
                            self.crystal_size_array_new, \
                            self.mcg_broken_additions, \
                            self.residue_additions, \
                            self.residue_count_additions, \
                            self.pcg_chem_residue_additions, \
                            self.mcg_chem_residue_additions, \
                            self.mass_balance, self.mcg_evolution

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
            self.residue_additions[step] = self.residue[step]

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
                vol_mcg = np.sum([self.volume_bins_medians_matrix * self.mcg])
                print("vol_mcg_total:", vol_mcg, "over", np.sum(self.mcg), "mcg")
                vol_residue = \
                    np.sum(self.residue_additions) + \
                    np.sum(self.pcg_chem_residue_additions) + \
                    np.sum(self.mcg_chem_residue_additions)

                print("mcg_intra_cb_residue_total:",
                      np.sum(self.residue_additions))
                print("pcg_chem_residue_total:",
                      np.sum(self.pcg_chem_residue_additions))
                print("mcg_chem_residue_total:",
                      np.sum(self.mcg_chem_residue_additions))
                print("vol_residue_total:", vol_residue)

                vol_pcg = self.calculate_vol_pcg()
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
            Newly formed list of the crystal sizes of the pcgs which are
            again represented by numpy arrays
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

        for i, (pcg, prob, csize, chem) in enumerate(
            zip(pcgs_old,
                interface_constant_prob_old,
                crystal_size_array_old,
                pcg_chem_weath_array_old)):

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
                probability_normalized = gen.normalize(prob)

            # Select interface to break pcg on
            interface = select_interface(i, probability_normalized, c)

            if self.enable_multi_pcg_breakage:
                prob_selected = probability_normalized[interface]
                print(prob_selected)
                interfaces_selected = \
                    np.where(probability_normalized > prob_selected)[0]
                print(interfaces_selected, interfaces_selected.size)

                pcg_new = np.array_split(pcg, interfaces_selected)
                csize_new = np.array_split(csize, interfaces_selected)
                chem_new = np.array_split(chem, interfaces_selected)
                prob_new = np.array_split(prob, interfaces_selected)

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
                    mcg_temp[chem[interface-1]][pcg[interface-1]]\
                        .append(csize[interface-1])

                # Evaluate second new pcg
                if pcg_length - interface != 1:
                    pcgs_new_append(pcg[interface:])
                    crystal_size_array_new_append(csize[interface:])
                    pcg_chem_weath_array_new_append(chem[interface:])
                    interface_constant_prob_new_append(prob[interface:])
                else:
                    mcg_temp[chem[interface]][pcg[interface]]\
                        .append(csize[interface])

                # Remove interface from interface_counts_matrix
                # Faster to work with matrix than with list and post-loop
                # operations as with the mcg counting
                self.interface_counts[pcg[interface-1], pcg[interface]] -= 1
                # interface_indices.append((pcg[interface-1], pcg[interface]))

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

        mcg_new = self.mcg.copy()
        mcg_new += mcg_temp_matrix

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

        residue_new = np.sum(residue_new, axis=0)
        residue_count_new = np.sum(residue_count_new, axis=0)

        return mcg_new, residue_new, residue_count_new

    def chemical_weathering_mcg(self, shift=1):
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

        for i, (pcg, prob, csize, chem_old) in enumerate(
            zip(self.pcgs_new,
                self.interface_constant_prob_new,
                self.crystal_size_array_new,
                self.pcg_chem_weath_array_new)):

            # Perform weathering
            chem = chem_old + shift

            # Redisue
            # 1. Residue from mcg being in a negative grain size class
            thresholds = self.negative_volume_thresholds[chem, pcg]
            remaining_crystals = np.where(csize >= thresholds)

            pcg_remaining = pcg[remaining_crystals]
            csize_remaining = csize[remaining_crystals]
            chem_remaining = chem[remaining_crystals]
            chem_old_remaining = chem_old[remaining_crystals]

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
                    gen.count_and_convert_interfaces_to_matrix(
                        pcg, self.n_minerals)

                new_interfaces = \
                    gen.count_and_convert_interfaces_to_matrix(
                        pcg_remaining, self.n_minerals)

                diff_interfaces = old_interfaces - new_interfaces

                interface_counts_matrix_diff += diff_interfaces

                interface_size_prob = \
                    gen.get_interface_size_prob(csize_remaining)
                interface_strength_prob = \
                    gen.get_interface_strengths_prob(
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
                gen.weighted_bin_count(pcg_dissolved,
                                       volumes_old_selected,
                                       self.n_minerals)

            # 2. Residue from material being weathered
            dissolved_volume_selected = \
                self.volume_change_matrix[chem_old_remaining,
                                          pcg_remaining,
                                          csize_remaining]
            residue_2 = \
                gen.weighted_bin_count(pcg_remaining,
                                       dissolved_volume_selected,
                                       self.n_minerals)

            # Add residue together per mineral
            residue_per_mineral += residue_1 + residue_2

        # Update interface_counts_matrix
        interface_counts_matrix_new = \
            self.interface_counts + interface_counts_matrix_diff

        return pcgs_new, crystal_size_array_new, interface_constant_prob_new, \
            pcg_chem_weath_array_new, residue_per_mineral, \
            interface_counts_matrix_new

    def chemical_weathering_pcg(self, shift=1):
        """Not taking into account that crystals on the inside of the
        pcg will be less, if even, affected by chemical weathering than
        those on the outside of the pcg"""

        residue_per_mineral = np.zeros(self.n_minerals, dtype=np.float64)

        pcg_lengths = np.array([len(pcg) for pcg in self.pcgs_new],
                               dtype=np.uint32)

        pcg_concat = np.concatenate(self.pcgs_new)
        csize_concat = np.concatenate(self.crystal_size_array_new)
        chem_concat_old = np.concatenate(self.pcg_chem_weath_array_new)

        chem_concat = chem_concat_old + 1

        thresholds_concat = \
            self.negative_volume_thresholds[chem_concat, pcg_concat]

        remaining_crystals = csize_concat >= thresholds_concat
        dissolved_crystals = np.where(csize_concat < thresholds_concat)

        pcg_remaining = pcg_concat[remaining_crystals]
        csize_remaining = csize_concat[remaining_crystals]
        chem_remaining = chem_concat[remaining_crystals]
        chem_old_remaining = chem_concat_old[remaining_crystals]

        pcg_filtered = \
            np.array_split(remaining_crystals, np.cumsum(pcg_lengths[:-1]))

        pcg_lengths_remaining = \
            np.array([len(pcg[pcg]) for pcg in pcg_filtered],
                     dtype=np.uint32)

        pcg_lengths_cumul = np.cumsum(pcg_lengths_remaining)

        zero_indices = np.where(pcg_lengths_remaining == 0)
        count_0 = zero_indices[0].size
        pcg_lengths_cumul_zero_deleted = np.delete(pcg_lengths_cumul,
                                                   zero_indices)

        pcg_remaining_list = \
            np.array_split(pcg_remaining, pcg_lengths_cumul[:-1])
        csize_remaining_list = \
            np.array_split(csize_remaining, pcg_lengths_cumul[:-1])
        chem_remaining_list = \
            np.array_split(chem_remaining, pcg_lengths_cumul[:-1])

        # Mcg accounting
        pcg_to_mcg = \
            pcg_remaining[pcg_lengths_cumul[pcg_lengths_remaining == 1] - 1]
        csize_to_mcg = \
            csize_remaining[pcg_lengths_cumul[pcg_lengths_remaining == 1] - 1]
        chem_to_mcg = \
            chem_remaining[pcg_lengths_cumul[pcg_lengths_remaining == 1] - 1]

        mcg_csize_unq, mcg_csize_ind, mcg_csize_cnt = \
            np.unique(csize_to_mcg, return_index=True, return_counts=True)

        self.mcg[chem_to_mcg[mcg_csize_ind],
                 pcg_to_mcg[mcg_csize_ind],
                 mcg_csize_unq] += mcg_csize_cnt.astype(np.uint32)

        # Interfaces counts
        pcg_concat_for_interfaces = \
            np.insert(pcg_remaining,
                      pcg_lengths_cumul[:-1].astype(np.int64),
                      self.n_minerals)

        interface_counts_matrix_new = \
            gen.count_and_convert_interfaces_to_matrix(
                pcg_concat_for_interfaces, self.n_minerals)

        # Interface probability calculations
        csize_concat_for_interfaces = \
            csize_remaining.copy().astype(np.int16)
        csize_concat_for_interfaces = \
            np.insert(csize_concat_for_interfaces,
                      pcg_lengths_cumul[:-1].astype(np.int64),
                      -1)

        interface_size_prob_concat = \
            gen.get_interface_size_prob(csize_concat_for_interfaces)
        interface_size_prob_concat = \
            interface_size_prob_concat[interface_size_prob_concat > 0]

        interface_strength_prob_concat = \
            gen.get_interface_strengths_prob(
                gen.expand_array(self.interface_proportions_normalized),
                pcg_concat_for_interfaces)
        interface_strength_prob_concat = \
            interface_strength_prob_concat[interface_strength_prob_concat > 0]

        prob_remaining = \
            interface_size_prob_concat / interface_strength_prob_concat

        prob_remaining_list = \
            np.array_split(
                prob_remaining, pcg_lengths_cumul_zero_deleted[:-1] -
                np.arange(1, len(pcg_remaining_list) - count_0))

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
            gen.weighted_bin_count(pcg_dissolved,
                                   volumes_old_selected,
                                   self.n_minerals)

        # 2. Residue from material being weathered
        dissolved_volume_selected = \
            self.volume_change_matrix[chem_old_remaining,
                                      pcg_remaining,
                                      csize_remaining]
        residue_2 = \
            gen.weighted_bin_count(pcg_remaining,
                                   dissolved_volume_selected,
                                   self.n_minerals)

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

        return pcgs_new, crystal_size_array_new, interface_constant_prob_new, \
            pcg_chem_weath_array_new, residue_per_mineral, \
            interface_counts_matrix_new

    def create_bins_matrix(self):
        """Create the matrix holding the arrays with bins which each
        represent the inital bin array minus x times the chemical
        weathering rate per mineral class.
        """

        size_bins_matrix = \
            np.array([[self.size_bins - x * self.chem_weath_rates[i]
                      for i in range(self.n_minerals)]
                     for x in range(self.n_timesteps)])

        volume_bins_matrix = gen.calculate_volume_sphere(size_bins_matrix)

        return size_bins_matrix, volume_bins_matrix

    def create_bins_medians_matrix(self):
        size_bins_medians_matrix = \
            gen.calculate_bins_medians(self.size_bins_matrix)
        volume_bins_medians_matrix = \
            gen.calculate_bins_medians(self.volume_bins_matrix)

        return size_bins_medians_matrix, volume_bins_medians_matrix

    def create_search_bins_matrix(self):
        search_size_bins_matrix = \
            np.array([[self.search_size_bins - x * self.chem_weath_rates[i]
                      for i in range(self.n_minerals)]
                     for x in range(self.n_timesteps)])

        search_volume_bins_matrix = \
            gen.calculate_volume_sphere(search_size_bins_matrix)

        return search_size_bins_matrix, search_volume_bins_matrix

    def create_search_bins_medians_matrix(self):
        search_size_bins_medians_matrix = \
            gen.calculate_bins_medians(self.search_size_bins_matrix)

        search_volume_bins_medians_matrix = \
            gen.calculate_bins_medians(self.search_volume_bins_matrix)

        return search_size_bins_medians_matrix, \
            search_volume_bins_medians_matrix

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
        intra_cb_breaks_matrix = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype='object')
        diffs_volumes_matrix = \
            np.zeros((self.n_timesteps, self.n_minerals), dtype='object')

        for n in range(self.n_timesteps):
            print(n)
            for m in range(self.n_minerals):
                intra_cb_breaks_array = \
                    np.zeros(
                        (self.n_bins_medians -
                         self.intra_cb_threshold_bin_matrix[n, m],
                         len(self.intra_cb_breaks)),
                        dtype=np.uint16)
                diffs_volumes_array = \
                    np.zeros(
                        (self.n_bins_medians -
                         self.intra_cb_threshold_bin_matrix[n, m],
                         len(self.intra_cb_breaks)),
                        dtype=np.float64)

                for i, b in \
                    enumerate(range(self.intra_cb_threshold_bin_matrix[n, m] +
                                    self.n_bins_medians,
                                    self.n_bins_medians*2)):
                    intra_cb_breaks_array[i], diffs_volumes_array[i] = \
                        determine_intra_cb_dict_array_version(
                            b, self.ratio_search_volume_bins_matrix[n, m],
                            max_n_values=len(self.intra_cb_breaks))

                intra_cb_breaks_matrix[n, m] = intra_cb_breaks_array
                diffs_volumes_matrix[n, m] = diffs_volumes_array

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

    Not worth it adding numba to this function
    """
    size, corr = divmod(a.size, 2)

    ranger = np.arange(size, 0, -1, dtype=np.uint32)

    chance = np.append(ranger, ranger[-2+corr::-1])

    return chance


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

    residue_count = 0
    residue_new = 0

    # 1. Select mcg
    if floor:
        # 1st time selection
        mcg_selected = np.floor(mcg_new * prob[mineral_nr]).astype(np.uint32)
    else:
        # 2nd time selection
        mcg_selected = np.ceil(mcg_new * prob[mineral_nr]).astype(np.uint32)

    # Sliced so that only the mcg above the intra_cb_threshold_bin are
    # affected; same reasoning in for loop below.
    mcg_new[intra_cb_threshold_bin:] -= mcg_selected[intra_cb_threshold_bin:]

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

        p1 = i + intra_cb_threshold_bin \
            - np.arange(1, breaker_counts.size+1)
        p2 = p1 - intra_cb_breaks_to_use

        mcg_new[p1] += breaker_counts
        mcg_new[p2] += breaker_counts

        residue_new += \
            np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] *
                   diffs_volumes_to_use * breaker_counts)

    return mcg_new, residue_new, residue_count


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


### OLD CODE ###
@nb.njit
def account_interfaces(interface_counts, interface_indices):
    for interface_index in interface_indices:
        interface_counts[interface_index] -= 1
    return interface_counts
