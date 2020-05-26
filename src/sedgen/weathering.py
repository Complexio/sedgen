import numpy as np

def mechanical_weathering_vectorized(steps, verbose=False, threshold=1e-8):
    """Threshold value is based on number of crystals that is expected to be 
    present in one cubic meter of parent rock which in turn is based on the average bulk crystal size
    
    Example:
    --------
    200 million crystals in 1 m³ means that a proportion of 0.5e-8 presents one crystal.
    When a proprtion is less than or equal to this threshold this would thus mean that 
    the pcg is no longer a pcg but has become a mcg.
    """
    composition_new = np.array([1])
    pcg_evolution = np.zeros(steps)
    mcg_evolution = np.zeros(steps)
    pcg_memory_usage = np.zeros(steps)
    single_crystals = 0

    for i in range(steps):
        composition_old = composition_new
        if verbose:
            print("composition_old", composition_old)
            print("composition_old_shape", composition_old.shape)
            
        # Obtain random fractions
        ab = np.random.random((2, composition_old.shape[0]))
        ab_norm = np.divide(ab, np.sum(ab, axis=0))
        if verbose:
            print("ab", ab)
            print("ab_norm", ab_norm)

        # Create new composition from old one and fractions
        composition_new = np.multiply(ab_norm.T, composition_old.reshape(-1, 1)).flatten()
        if verbose:
            print("composition_new", composition_new)
            print("composition_new_sum", np.sum(composition_new), "\n")
            
        # Move fractions smaller than threshold to single crystals count
        condition = composition_new <= threshold
        single_crystals += np.sum(condition)
        if verbose:
            print("single_crystals", single_crystals)
        composition_new = composition_new[~condition]
        
        if verbose:
            print("composition_new", composition_new)
            print("composition_new_sum", np.sum(composition_new), "\n")
        
        # Keep track of evolution of pcg and mcg
        pcg_evolution[i] = composition_new.shape[0]
        mcg_evolution[i] = single_crystals
        pcg_memory_usage[i] = composition_new.nbytes / 1024**2
        if composition_new.shape[0] == 0:
            print(i)
            break
        
    return composition_new, single_crystals, pcg_evolution, mcg_evolution, pcg_memory_usage


def mechanical_weathering_absolute_vectorized(steps, verbose=False, threshold=1, start_nr=200e6):
    """Threshold value is based on number of crystals that is expected to be 
    present in one cubic meter of parent rock which in turn is based on the average bulk crystal size
    
    Example:
    --------
    200 million crystals in 1 m³ means that a proportion of 0.5e-8 presents one crystal.
    When a proprtion is less than or equal to this threshold this would thus mean that 
    the pcg is no longer a pcg but has become a mcg.
    """
    composition = np.array([start_nr], dtype='uint32')
    pcg_evolution = np.zeros(steps, dtype='uint32')
    mcg_evolution = np.array([start_nr]*steps, dtype='uint32')
    pcg_memory_usage = np.zeros(steps, dtype='float64')
    single_crystals = 0

    for i in range(steps):
#         composition_old = composition_new
        if verbose:
            print("composition_old", composition)
            print("composition_old_shape", composition.shape)
            
        # Obtain random fractions to divide pcgs in two parts
        a_prop = np.random.random(composition.shape[0])
#         ab_norm = np.divide(ab, np.sum(ab, axis=0))
        if verbose:
            print("a_prop", a_prop)
#             print("ab_norm", ab_norm)

        # Create new composition from old one and fractions
        a = np.multiply(a_prop, composition).astype('uint32')
#         b = composition - a
        if verbose:
            print("a", a)
            print("b", composition - a)
        composition = np.array((a, composition - a))
        if verbose:
            print("composition_new", composition)
            print("composition_new_sum", np.sum(composition), "\n")
            
        # Move fractions smaller than threshold to single crystals count
        single_crystals += np.sum(composition == threshold)
        if verbose:
            print("single_crystals", single_crystals)
        # Remove single crystals and zero values from pcg array
        composition = composition[composition > threshold]
        
        if verbose:
            print("composition_new", composition)
            print("composition_new_sum", np.sum(composition), "\n")
        
        # Keep track of evolution of pcg and mcg numbers...
        pcg_evolution[i] = composition.shape[0]
        mcg_evolution[i] = single_crystals
        # ... and pcg_array memory usage
        pcg_memory_usage[i] = composition.nbytes / 1024**2
        
        # Break out of for loop if there are no more pcgs left
        if composition.shape[0] == 0:
            print(i)
            break
        
    return composition, single_crystals, pcg_evolution, mcg_evolution, pcg_memory_usage