import numpy as np


class ModelEvolutionMixin:
    def __init__(self):
        self.pcg_crystals = [self.pr_crystals.copy()]
        self.pcg_interface_constant_prob = \
            [self.pr_interface_constant_prob.copy()]
        self.pcg_crystal_sizes = [self.pr_crystal_sizes.copy()]
        self.pcg_chem_weath_states = \
            [np.zeros_like(self.pr_crystals.copy())]

        self.mcg = \
            np.zeros((self.n_steps, self.pr_n_minerals, self.n_bins),
                     dtype=np.uint64)
        self.residue = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype=np.float64)
        self.residue_count = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype=np.uint32)

        # Model's evolution tracking arrays initialization
        self.mcg_chem_residue = 0
        self.pcg_chem_residue = 0

        self.pcg_additions = np.zeros(self.n_steps, dtype=np.uint32)
        self.mcg_additions = np.zeros(self.n_steps, dtype=np.uint64)
        self.mcg_broken_additions = np.zeros(self.n_steps, dtype=np.uint64)
        self.residue_additions = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype=np.float64)
        self.residue_count_additions = \
            np.zeros(self.n_steps, dtype=np.uint32)
        self.pcg_chem_residue_additions = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype=np.float64)
        self.mcg_chem_residue_additions = \
            np.zeros((self.n_steps, self.pr_n_minerals), dtype=np.float64)

        self.pcg_comp_evolution = []
        self.pcg_size_evolution = []

        self.mcg_evolution = \
            np.zeros((self.n_steps, self.pr_n_minerals, self.n_bins),
                     dtype=np.uint64)

        self.vol_mcg_evolution = np.zeros(self.n_steps, dtype=np.float64)
        self.vol_pcg_evolution = np.zeros(self.n_steps, dtype=np.float64)
        self.vol_residue_evolution = np.zeros(self.n_steps, dtype=np.float64)
        self.mass_balance = np.zeros(self.n_steps, dtype=np.float64)

        self.new_material_volumes = np.zeros(self.n_steps, dtype=np.float64)
