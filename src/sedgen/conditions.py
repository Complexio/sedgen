# This script should contain default boundary conditions such as:
#     - Temperature
#     - pH
#     - Precipitation
#     - Relief
# for different environments.

# These boundary conditions will mainly have an effect on the mechanical
# and chemical weathering rates and therefore indirectly on the
# mechanical vs chemical weathering ratio.

# The possibility to add new environments by the user could also be
# taken up at a later stage.
import numpy as np


class BoundaryConditionsMixin:
    def __init__(self):
        self.chem_weath_rates = self.set_chem_weath_rates()
        self.tectonic_regimes = self.set_tectonic_regimes()

    def set_chem_weath_rates(self):
        # Rows represent mineral classes, columns represent climate classes
        chem_weath_rates = \
            np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                      [1.5, 1.5, 1.5, 1.5, 1.5],
                      [3.0, 3.0, 3.0, 3.0, 3.0],
                      [2.0, 2.0, 2.0, 2.0, 2.0],
                      [0.8, 0.8, 0.8, 0.8, 0.8],
                      [1.6, 1.6, 1.6, 1.6, 1.6]])

        return chem_weath_rates

    def set_tectonic_regimes(self):
        tectonic_regimes = \
            np.array([])

        return tectonic_regimes
