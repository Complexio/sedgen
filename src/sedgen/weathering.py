import numpy as np
import numba as nb

from sedgen import sedgen

def inter_crystal_breakage(model, n_timesteps, verbose=False, random_seed=343, n_standard_cases=2000, p=[0.5], intra_cb_thresholds=[1/256]):

    prng = np.random.RandomState(random_seed)

    interface_constant_prob = \
        model.interface_size_prob * model.interface_strengths_prob
    standard_prob_loc_cases = np.array([create_interface_location_prob(np.arange(x)) for x in range(1, n_standard_cases+1)], dtype=np.object)

    bins = model.volume_bins_medians.copy()
    volume_perc_change_unit = bins[0] / bins[1]
    n_minerals = len(model.minerals)
    n_bins = len(bins)

    pcgs_new = [model.interface_array.copy()]
    interface_constant_prob_new = [interface_constant_prob.copy()]
    mass_balance_initial = np.sum(model.simulated_volume)
#     print("mass balance initial:", mass_balance_initial)
    crystal_size_array_new = [model.crystal_size_array.copy()]
    interface_counts = model.interface_counts_matrix.copy()

    mcg = np.zeros((n_minerals, n_bins), dtype=np.uint32)
#     residue_mcg_total = np.zeros(n_minerals, dtype=np.float64)
    residue = np.zeros(n_minerals, dtype=np.float64)
    residue_count = np.zeros(n_minerals, dtype=np.uint32)

    # Create array of intra-crystal brekage probabilities
    if len(p) == 1:
        p = np.array([p] * n_minerals)
    elif len(p) == n_minerals:
        p = np.array(p)
    else:
        raise ValueError("p should be of length 1 or same length as minerals")

    # Create array of intra-cyrstal breakage size thresholds
    if len(intra_cb_thresholds) == 1:
        intra_cb_thresholds = np.array([intra_cb_thresholds] * n_minerals)
    elif len(intra_cb_thresholds) == n_minerals:
        intra_cb_thresholds = np.array(intra_cb_thresholds)
    else:
        raise ValueError("intra_cb_thresholds should be of length 1 or same length as minerals")

    pcg_additions = np.zeros(n_timesteps, dtype=np.uint32)
    mcg_additions = np.zeros(n_timesteps, dtype=np.uint32)
    mcg_broken_additions = np.zeros(n_timesteps, dtype=np.uint32)
    residue_additions = np.zeros(n_timesteps, dtype=np.float64)
    residue_count_additions = np.zeros(n_timesteps, dtype=np.uint32)
    pcg_chem_residue_additions = np.zeros((n_timesteps, n_minerals), dtype=np.float64)
    mcg_chem_residue_additions = np.zeros((n_timesteps, n_minerals), dtype=np.float64)

    pcg_comp_evolution = []
    pcg_size_evolution = []

    intra_cb_dict = \
        determine_intra_cb_dict(2800, model.ratio_search_bins)
    diffs_volumes = intra_cb_dict[2].copy()
    intra_cb_breaks = intra_cb_dict[1].copy()

    for step in range(n_timesteps):
        print(f"{step}/{n_timesteps}", end="\r", flush=True)

        # intra-crystal breakage
        mcg_broken, residue, residue_count = intra_crystal_breakage_binned(mcg, p, intra_cb_thresholds, residue, residue_count, model,
             intra_cb_breaks, diffs_volumes)
#         print("after intra_cb mcg_vol:", np.sum(bins * mcg_broken))
#         print("after intra_cb residue:", np.sum(residue))
        mcg_broken_additions[step] = np.sum([np.sum(x) for x in mcg_broken])# - np.sum(mcg_broken_additions)
#         residue_mcg_total += residue
        residue_additions[step] = np.sum(residue) - np.sum(residue_additions)
        residue_count_additions[step] = np.sum(residue_count)
        mcg = mcg_broken.copy()

        pcgs_old = pcgs_new
        pcgs_new = []
        pcgs_new_append = pcgs_new.append

        interface_constant_prob_old = interface_constant_prob_new
        interface_constant_prob_new = []
        interface_constant_prob_new_append = interface_constant_prob_new.append

        crystal_size_array_old = crystal_size_array_new
        crystal_size_array_new = []
        crystal_size_array_new_append = crystal_size_array_new.append

        c_creator = np.random.RandomState(step)
        c = c_creator.random(size=pcg_additions[step-1] + 1)

        mcg_temp = [[] for i in range(n_minerals)]
#         interface_indices = List()

        for i, (pcg, prob, csize) in enumerate(zip(pcgs_old, interface_constant_prob_old, crystal_size_array_old)):

            # Select interface for inter-crystal breakage
#             try:
            if len(pcg) <= n_standard_cases:
                location_prob = standard_prob_loc_cases[len(pcg) - 1]
#             except IndexError:
            else:
                location_prob = create_interface_location_prob(pcg)

            # Calculate normalized probability
            probability_normalized = \
                calculate_normalized_probability(location_prob, prob)

            # Select interface to break pcg on
            interface = select_interface(i, probability_normalized, c)

            # Using indexing instead of np.split is faster.
            # Also avoids the problem of possible 2D arrays instead of
            # 1D being created if array gets split in half.
            # Evuluate first new pcg
            if pcg[:interface].size != 1:  # This implies that len(new_prob) != 0
                pcgs_new_append(pcg[:interface])
                crystal_size_array_new_append(csize[:interface])
                interface_constant_prob_new_append(prob[:interface-1])
            else:
                mcg_temp[pcg[interface-1]].append(csize[interface-1])

            # Evaluate second new pcg
            if pcg[interface:].size != 1:  # This implies that len(new_prob) != 0
                pcgs_new_append(pcg[interface:])
                crystal_size_array_new_append(csize[interface:])
                interface_constant_prob_new_append(prob[interface:])
            else:
                mcg_temp[pcg[interface]].append(csize[interface])

            # Remove interface from interface_counts_matrix
            # Faster to work with matrix than with list and post-loop
            # operations as with the mcg counting
            interface_counts[pcg[interface-1], pcg[interface]] -= 1
#             interface_indices.append((pcg[interface-1], pcg[interface]))

        # Add counts from mcg_temp to mcg
#         mcg_temp_matrix = np.zeros((n_minerals, n_bins), dtype=np.uint32)
        mcg_temp_matrix = np.asarray([np.bincount(mcg_temp_list, minlength=n_bins) for mcg_temp_list in mcg_temp])
#         print(mcg_temp_matrix.shape)
#         for i, mcg_bin_count in enumerate(mcg_bin_counts):
#             mcg_temp_matrix[i, :len(mcg_bin_count)] = mcg_bin_count
        mcg += mcg_temp_matrix.astype(np.uint32)

        # Remove counts from interface_counts via interface_temp
#         interface_temp_count = count_interfaces(np.array(interface_temp))
#         interface_temp_matrix = convert_counted_interfaces_to_matrix(interface_temp_count)
#         interface_counts -= interface_temp_matrix
#         interface_counts = account_interfaces(interface_counts, interface_indices)


        # To Do: Provide option for different speeds of chemical
        # weathering per mineral class. This could be done by moving to
        # a different number of volume bins (n) per mineral class.
        # For the volume_perc_change this would become:
        # volume_perc_change = volume_perc_change ** n

        # Chemical weathering pcg
        crystal_size_array_new, pcg_chem_residue = chemical_weathering_pcg(crystal_size_array_new, pcgs_new, bins, volume_perc_change_unit)
        pcg_chem_residue_additions[step] = pcg_chem_residue

        # Chemical weathering mcg
        mcg, mcg_chem_residue = chemical_weathering_mcg(mcg, volume_perc_change_unit, bins)
        mcg_chem_residue_additions[step] = mcg_chem_residue

        pcg_additions[step] = len(pcgs_new)
#         print("pcg_nr:", pcg_additions[step])
#         n_mcg = 0
#         for m in mcg:
#             n_mcg += len(m)
        mcg_additions[step] = np.sum(mcg)# - np.sum(mcg_additions)
#         print("mcg_nr", mcg_additions[step])
#         print("mcg_broken_nr", mcg_broken_additions[step])
#         mcg_additions[step] = n_mcg - np.sum(mcg_additions)

        pcg_comp_evolution.append(pcgs_new)
        pcg_size_evolution.append(crystal_size_array_new)

#         # Mass balance check
#         # mass balance = vol_pcg + vol_mcg + residue
#         vol_mcg = np.sum([bins * mcg])
#         print("vol_mcg:", vol_mcg)
#         vol_residue = \
#             np.sum(residue_additions) + np.sum(pcg_chem_residue_additions) + \
#             np.sum(mcg_chem_residue_additions)
#         print("mcg_intra_cb_residue:", np.sum(residue_additions), np.sum(residue_mcg_total))
#         print("pcg_chem_residue:", np.sum(pcg_chem_residue_additions))
#         print("mcg_chem_residue:", np.sum(mcg_chem_residue_additions))
#         print("vol_residue:", vol_residue)
#         vol_pcg = np.sum([np.sum(bins[pcg]) for pcg in crystal_size_array_new])
#         print("vol_pcg:", vol_pcg)

#         mass_balance = vol_pcg + vol_mcg + vol_residue
#         print(f"new mass balance at step {step}: {mass_balance}\n")

        if not pcgs_new: # Faster to check if pcgs_new has any items
            print(f"After {step} steps all pcg have been broken down to mcg")
            break

    return pcgs_new, mcg, pcg_additions, mcg_additions, pcg_comp_evolution, \
        pcg_size_evolution, interface_counts, crystal_size_array_new, \
        mcg_broken_additions, residue_additions, residue_count_additions, \
        pcg_chem_residue_additions, mcg_chem_residue_additions


def create_interface_location_prob(a):
    """Creates an array descending and then ascending again to represent
    chance of inter crystal breakage of a poly crystalline grain (pcg).
    The outer interfaces have a higher chance of breakage than the
    inner ones based on their location within the pcg.
    This represents a linear function.
    Perhaps other functions might be added (spherical) to see the effects later on
    """
    size, corr = divmod(len(a), 2)
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
    # This also results in that if the final crystal is selected,
    # the pcg will not be broken in two.

    return interface


# Speedup from 2m45s to 1m30s
# Parallelizable but not performant for small pcg
@nb.njit(cache=True)
def calculate_normalized_probability(location_prob, prob):
    probability = location_prob * prob
    return probability / np.sum(probability)


def intra_crystal_breakage_binned(mcg_old, p, intra_cb_thresholds, residue_old, residue_count_old, model, intra_cb_breaks, diffs_volumes):
    mcg_new = np.zeros_like(mcg_old)
    residue_new = residue_old.copy()
    residue_count_new = residue_count_old.copy()

    for i, m_old in enumerate(mcg_old):
        if all(m_old == 0):
            mcg_new[i] = m_old
        else:
#             m_new, residue_new, residue_count_new = perform_intra_crystal_breakage_binned(m_old, p, intra_cb_thresholds, i)
            m_new, residue_add, residue_count_add = perform_intra_crystal_breakage_2d(m_old, p, i, model.search_bins_medians, intra_cb_breaks, 200, diffs_volumes)
            mcg_new[i] = m_new
            residue_new[i] += residue_add
            residue_count_new[i] += residue_count_add
    return mcg_new, residue_new, residue_count_new


# Enable option for chemical weathering on not all mcg at same time
# later on (if this is realistic actual that some mcg will be affected
# by chemical weathering while others are not).


@nb.njit(cache=True)
def perform_intra_crystal_breakage_2d(mcg_old, prob, i, search_bins, intra_cb_breaks, intra_cb_threshold_bin, diffs_volumes, floor=True, verbose=False, corr=1):
    # Certain percentage of mcg has to be selected for intra_cb
    # Since mcg are already binned it doesn't matter which mcg get
    # selected in a certain bin, only how many

    mcg_new = mcg_old.copy()
    n_bins = len(mcg_new)
#     print("pre-intra_cb volume:", np.sum(search_bins[-1500:] * mcg_new))
    residue_count = 0
    residue_new = 0

    # 1. Select mcg
    if floor:
        # 1st time selection
        mcg_selected = np.floor(mcg_new * prob[i]).astype(np.uint32)
    else:
        # 2nd time selection
        mcg_selected = np.ceil(mcg_new * prob[i]).astype(np.uint32)

    # Sliced so that only the mcg above the intra_cb_threshold_bin are
    # affected; same reasoning in for loop below.
    mcg_new[intra_cb_threshold_bin:] -= mcg_selected[intra_cb_threshold_bin:]

    # 2. Create break points
    for i, n in enumerate(mcg_selected[intra_cb_threshold_bin:]):
        breaker = np.random.randint(low=1, high=len(intra_cb_breaks), size=n).astype(np.uint16)
        p1 = i + intra_cb_threshold_bin + n_bins - breaker - corr
        p2 = p1 - intra_cb_breaks[breaker]
        if verbose and len(p1) != 0:
            print(i, intra_cb_threshold_bin, n_bins, breaker)
            print(p1)
            print(p2)
        p = np.concatenate((p1, p2))
        p_filtered = p[p >= n_bins]
        if len(p_filtered) != 0:
            p_count = np.bincount(p_filtered - n_bins).astype(np.uint32)
            mcg_new[:len(p_count)] += p_count
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
        if intra_cb_breaks[0] > intra_cb_threshold_bin:
            residue = p[p < n_bins]
            residue_count += len(residue)
            residue_new += np.sum(search_bins[residue])

        # Addition of small fraction of material that
        # gets 'lost' during intra_cb_breakage to residue.
        if verbose and len(p1) != 0:
            print(search_bins[i + intra_cb_threshold_bin + n_bins])
            print(diffs_volumes[breaker])
        residue_new += np.sum(search_bins[i + intra_cb_threshold_bin + n_bins] * diffs_volumes[breaker])

#     print("post-intra_cb volume:", np.sum(search_bins[-1500:] * mcg_new) + residue_new)
#     print("residue_new:", residue_new)
    return mcg_new, residue_new, residue_count


# Perform function per mineral class
def chemical_weathering_mcg(mcg_old, volume_perc_change_unit, bins, shift=1):
    mcg_new = np.zeros_like(mcg_old)
    volume_perc_change = volume_perc_change_unit ** shift

    ## Create selection array based on chemical weathering importance

    ## Select mcg
    ## Keep non-selected mcg in their respective size/volume bins

    residue_per_mineral = calculate_mcg_chem_residue(mcg_old, bins, volume_perc_change)

    # Keep track of residue formed by shifting bins
#     residue_per_mineral = np.sum(mcg_old[:, 1:] * volume_bins_medians[1:], axis=1) * (1 - volume_perc_change)
    # Add total volume of first bin to residue
#     residue_per_mineral += mcg_old[:, 0] * volume_bins_medians[0]

    # Reduce size/volume of selected mcg by decreasing their size/volume bin by one
    mcg_new = np.roll(mcg_old, shift=-shift, axis=1)
    # Remove artefact from roll operation
    mcg_new[:, -shift:] = 0
#     print(mcg_new)

    ## Concatenate non affected mcg with chemical weathered mcg

    return mcg_new, residue_per_mineral


@nb.njit(cache=True)
def calculate_mcg_chem_residue(mcg_old, bins, volume_perc_change):
    # Keep track of residue formed by shifting bins
    residue_per_mineral = np.sum(mcg_old[:, 1:] * bins[1:], axis=1) * (1 - volume_perc_change)
    # Add total volume of first bin to residue
    residue_per_mineral += mcg_old[:, 0] * bins[0]

    return residue_per_mineral


# Perform function on all pcgs in csize_array simultaneously

def chemical_weathering_pcg(csize_old, pcg_old, bins, volume_perc_change_unit, shift=1):
    """Not taking into account that crystals on the inside of the pcg
    will be less, if even, affected by chemical weathering than those on
    the outside of the pcg"""
    # Need to keep structure of pcgs so can't concatenate here
    csize_new = np.subtract(csize_old, shift, dtype='object')
    volume_perc_change = volume_perc_change_unit ** shift

    # Also need to keep track of formed residue
    modal_mineralogy, volumes_old = sedgen.calculate_modal_mineralogy_pcg(pcg_old, csize_old, bins)

    old_volume = np.sum(volumes_old)
    residue = old_volume * (1 - volume_perc_change)

    residue_per_mineral = residue * modal_mineralogy

    return csize_new, residue_per_mineral


def determine_intra_cb_dict(bin_label, ratio_search_bins, verbose=False, corr=1):
    intra_cb_dict = {}
    diffs = []
    diffs_volumes = []

    specific_ratios = ratio_search_bins[:bin_label] / ratio_search_bins[bin_label]

    for i in range(bin_label-1, 0, -1):
        y = 1 - specific_ratios[i]
        # Minus 1 for found bin so that volume sum of two new mcg
        # is bit less than 100%; remainder goes to residue later on.
        found_bin = np.argmax(y < specific_ratios) - corr
        if verbose:
            print(i, specific_ratios[i], y, found_bin, i - found_bin)
            print(specific_ratios[i] + specific_ratios[found_bin])
        intra_cb_dict[i] = found_bin
        diffs.append(i - found_bin)
        diffs_volumes.append(1 - (specific_ratios[i] + specific_ratios[found_bin]))
        if i - found_bin == 0 + corr:
            break
    return intra_cb_dict, np.array(diffs), np.array(diffs_volumes)



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
