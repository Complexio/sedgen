import matplotlib.pyplot as plt
import seaborn as sns
import ternary
# import plotly

import numpy as np

from sedgen import general as gen
# from sedgen.initialization import SedGen

phi_classes = list(range(10, -6, -1))


class SedGenEvolution():
    """Initialises the data needed to plot a sedgen model's evolution
    thoughout a predetermined range of model steps

    Parameters:
    -----------
    model : SedGen
        sedgen model to base calculations on
    start_step : int (optional)
        First step to start range of steps from to perform calculations
        on; defaults to 0.
    end_step : int (optional)
        Final step in range of steps to perform calculations on.
        Note that this step is excluded in the generated range.
        Defaults to 100.
    step_interval : int (optional)
        Interval of steps to perform calculations on; defaults to 5
    """

    def __init__(self, model, start_step=0, end_step=100, step_interval=5):
        self.model = model
        self.start_step = start_step
        self.end_step = end_step
        self.step_interval = step_interval
        self.steps = \
            list(range(self.start_step, self.end_step, self.step_interval))
        self.n_steps = len(self.steps)
        self.steps_to_run = \
            {k: v for k, v in zip(range(self.n_steps), self.steps)}

        print(f"Starting calculation of volumes' evolution of model from step "
              f"{self.start_step} to step {self.end_step} with a "
              f"{self.step_interval} step(s) interval.")
        print(self.steps)
        print("Calculating pcg_volumes_evolution of sedgen model...")
        self.pcg_volumes_per_phi = \
            self.calculate_grouped_pcg_volumes()
        print("Calculating mcg_volumes_evolution of sedgen model...")
        self.mcg_volumes_per_phi = \
            self.calculate_grouped_mcg_volumes()
        print("Calculating residue_volumes_evolution of sedgen model...")
        self.residue_volumes_evolution = self.calculate_residue_volume()
        print("Done.")

    def calculate_grouped_pcg_volumes(self):

        pcg_volumes_per_phi = np.zeros((self.n_steps, 6, 15))
        whole_phi_classes = np.arange(0, 1501, 100).astype(np.int64)
    #     whole_phi_classes[0] = 0

        for n, step in self.steps_to_run.items():
    #         total_crystals = 0
            for m in range(6):
                print(step, m, end="\r", flush=True)
    #             whole_phi_classes_true = model.volume_bins_medians_matrix[n][m][whole_phi_classes]

    #             absolute_volumes_grouped_total = np.zeros(16)
                pcg_crystal_sizes_filtered_grouped_total = np.zeros(1500)

                try:
                    for pcg_crystals, pcg_crystal_sizes, pcg_chem_weath_states \
                        in zip(self.model.pcg_crystals_evolution[step],
                               self.model.pcg_crystal_sizes_evolution[step],
                               self.model.pcg_chem_weath_states_evolution[step]):
                        # Filter crystal size and chemical weathering state
                        # arrays by mineral class for selected pcg
                        mineral_filter = pcg_crystals == m
                        pcg_crystal_sizes_filtered = pcg_crystal_sizes[mineral_filter]
                        pcg_chem_weath_states_filtered = \
                            pcg_chem_weath_states[mineral_filter]
    #                     total_crystals += len(pcg_crystal_sizes_filtered)

                        # Get absolute volumes array
                        absolute_volumes = self.model.volume_bins_medians_matrix[
                            pcg_chem_weath_states_filtered, m,
                            pcg_crystal_sizes_filtered]

                        pcg_crystal_sizes_filtered_grouped = gen.weighted_bin_count(pcg_crystal_sizes_filtered, w=absolute_volumes, ml=1500)
                        pcg_crystal_sizes_filtered_grouped_total += \
                            pcg_crystal_sizes_filtered_grouped

                    # Attach absolute volume to grouped whole phi classes array
                    absolute_volumes_grouped = \
                        np.array([
                            np.sum(pcg_crystal_sizes_filtered_grouped_total[
                                x:x+100])
                            for x in whole_phi_classes[:-1]])

                except IndexError:
                    print(f"All {n} steps accounted for.")
                    return pcg_volumes_per_phi

                pcg_volumes_per_phi[n, m] = absolute_volumes_grouped
    #         print(np.sum(pcg_volumes_per_phi[n]))
    #         print(total_crystals)

        return pcg_volumes_per_phi

    def calculate_grouped_mcg_volumes(self):
        mcg_volumes_per_phi = np.zeros((self.n_steps, 6, 15))
        whole_phi_classes = np.arange(0, 1501, 100).astype(np.int64)
    #     print(whole_phi_classes)

        # For each model step we need to collect the volumes of all mcg
        # crystals per phi class
        for n, step in self.steps_to_run.items():
    #         total_crystals = 0

            print(step, end="\r", flush=True)

            mcg_of_interest = self.model.mcg_evolution[step, :, :, :]

            # Firstly, the mcg numbers need to be multiplied with the volume
            # bin medians to obtain mcg volumes
            absolute_volumes = self.model.volume_bins_medians_matrix * mcg_of_interest
    #         print(absolute_volumes.shape)
    #         print(np.sum(absolute_volumes))

            # Secondly, these volumes need to be summed per whole phi
            # class
            absolute_volumes_grouped = np.array([np.sum(absolute_volumes[:, :, x:x+100], axis=2) for x in whole_phi_classes[:-1]])
    #         print(absolute_volumes_grouped.shape)
            absolute_volumes_grouped = np.sum(absolute_volumes_grouped, axis=1)
    #         print(absolute_volumes_grouped.shape)

            mcg_volumes_per_phi[n] = absolute_volumes_grouped.T

        return mcg_volumes_per_phi

    def calculate_residue_volume(self):
        return np.array([self.model.vol_residue_evolution[x]
                        for x in range(self.start_step,
                                       self.end_step,
                                       self.step_interval)])

    def calculate_QFR_data(self):
        """Calculates the data needed for a mcg_Q, mcg_P+K, pcg_all
        ternary diagram.
        """
        # A = mcg_Q
        Y = self.mcg_volumes_per_phi[:, 0, :]
        # B = mcg_P+K
        Z = self.mcg_volumes_per_phi[:, 1, :] + \
            self.mcg_volumes_per_phi[:, 2, :]
        # C = pcg_all
        X = np.sum(self.pcg_volumes_per_phi, axis=1)

        T = np.stack([X, Y, Z])
        T = np.moveaxis(T, [0, 1, 2], [2, 1, 0])
        # Ignoring divide by zero and invalid dividing since we expect
        # some values to be nan.
        with np.errstate(divide='ignore', invalid='ignore'):
            T_norm = T / np.sum(T, axis=-1)[:, :, np.newaxis]

        return T_norm

    def calculate_QFOth_data(self):
        """Calculates the data needed for a pcg_Q, pcg_P+K, pcg_Oth
        ternary diagram.
        """
        # A = pcg_Q
        Y = self.pcg_volumes_per_phi[:, 0, :]
        # B = pcg_P+K
        Z = self.pcg_volumes_per_phi[:, 1, :] + \
            self.pcg_volumes_per_phi[:, 2, :]
        # C = pcg_all
        X = np.sum(self.pcg_volumes_per_phi[:, 3:, :], axis=1)

        T = np.stack([X, Y, Z])
        T = np.moveaxis(T, [0, 1, 2], [2, 1, 0])
        # Ignoring divide by zero and invalid dividing since we expect
        # some values to be nan.
        with np.errstate(divide='ignore', invalid='ignore'):
            T_norm = T / np.sum(T, axis=-1)[:, :, np.newaxis]

        return T_norm

    def QFR_ternary_plot(self, selected_phi_classes=None,
                         save_filename=None):
        """Ternary diagram plot of mcg's and pcg's modal mineralogy with
        trajectory along the model's steps.
        A=Y=mcg_Q, B=Z=mcg_PK, C=X=pcg_QPKBOA.
        Data has been grouped by whole phi grain size classes.

        Parameters:
        -----------
        selected_phi_classes : None or list (optional)
            List of phi classes to plot; defaults to None so that all
            phi classes are plotted.
        save_filename : str (optional)
            Name to use for saving figure; defaults to None so that no
            figure is saved.

        """
        # Calculate data
        T_norm = self.calculate_QFR_data()

        # Filter data
        if selected_phi_classes is None:
            selected_phi_classes = phi_classes

        # Plot data
        fig, ax = plt.subplots(figsize=(12, 10.8))
        figure, tax = ternary.figure(ax=ax, scale=1.0)
        tax.boundary()
        tax.gridlines(multiple=0.2, color="black")
        # tax.set_title("QFR plot", fontsize=20)
        fontsize = 12
        tax.right_corner_label("Rock Fragments", fontsize=fontsize)
        tax.top_corner_label("Quartz", fontsize=fontsize)
        tax.left_corner_label("Feldspar", fontsize=fontsize)
        points = []
        # Get data
        for p, phi in enumerate(T_norm):
            if phi_classes[p] in selected_phi_classes:
                points = phi
                # Plot the data
                tax.plot(points, linewidth=2.0, label=f"{phi_classes[p]}_phi",
                         marker='o', color=plt.cm.tab20(p))
        tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
        tax.legend(loc=1)
        plt.axis('off')
        # plt.tight_layout()
        if save_filename:
            tax.savefig(f"QFR_ternary_plot_{save_filename}.pdf")
        tax.show()

    def QFOth_ternary_plot(self, selected_phi_classes=None,
                           save_filename=None):
        """Ternary diagram plot of pcg's modal mineralogy with trajectory
        along the model's steps. A=pcg_Q, B=pcg_PK, C=pcg_BOA.
        Data has been grouped by whole phi grain size classes.

        Parameters:
        -----------
        selected_phi_classes : None or list (optional)
            List of phi classes to plot; defaults to None so that all
            phi classes are plotted.
        save_filename : str (optional)
            Name to use for saving figure; defaults to None so that no
            figure is saved.

        """
        # Calculate data
        T_norm = self.calculate_QFOth_data()

        # Filter data
        if selected_phi_classes is None:
            selected_phi_classes = phi_classes

        # Plot data
        fig, ax = plt.subplots(figsize=(12, 10.8))
        figure, tax = ternary.figure(ax=ax, scale=1.0)
        tax.boundary()
        tax.gridlines(multiple=0.2, color="black")
        # tax.set_title("QFR plot", fontsize=20)
        fontsize = 12
        tax.right_corner_label("Others", fontsize=fontsize)
        tax.top_corner_label("Quartz", fontsize=fontsize)
        tax.left_corner_label("Feldspar", fontsize=fontsize)
        points = []
        # Get data
        for p, phi in enumerate(T_norm):
            if phi_classes[p] in selected_phi_classes:
                points = phi
                # Plot the data
                tax.plot(points, linewidth=2.0, label=f"{phi_classes[p]}_phi",
                         marker='o', color=plt.cm.tab20(p))
        tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
        tax.legend(loc=1)
        plt.axis('off')
        # plt.tight_layout()
        if save_filename:
            tax.savefig(f"QFO_ternary_plot_{save_filename}.pdf")
        tax.show()


def bulk_grain_size_barplot(model, step):
    """Bar plot of the bulk grain size, meaning grain size in both mcg
    and pcg combined.

    Parameters:
    -----------

    """
    pass


def solids_vs_residue_lineplot(model):
    """Line plot of the rock material (either parent rock or sediment)
    versus the residue material as an evolution throughout the model's
    steps.

    Parameters:
    -----------

    """
    pass


def lineplotpcgmcg(pluton, sedgenmech, name):
    fig, ax = plt.subplots()
    ax.plot(np.log10(sedgenmech.pcg_additions), label="pcg", lw=2)
    ax.plot(np.log10(sedgenmech.mcg_additions), label="mcg", lw=2)

    # ax.text(70, 12, "mcg", color=sns.color_palette()[1], fontsize=12)
    # ax.text(70, 2.5, "pcg", color=sns.color_palette()[0], fontsize=12)

    # Set the color of the visible spines
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    # ax.set_xlim(0, 77)
    # ax.spines['bottom'].set_bounds(0, 77)
    # ax.spines['left'].set_bounds(0, 14)
    ax.spines['left'].set_position(('outward', 10))
    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='out',
                   colors='grey',
                   labelsize=9)
    ax.set_title('pcg,mcg'+ pluton)
    ax.set_xlabel("Timesteps", color='grey')
    ax.set_ylabel("$\log_{10}$n", rotation=0, labelpad=20, color='grey')
    # Set facecolor of figure
    plt.gcf().set_facecolor('white')
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig('lineplot_pcgmcg_evolution_chem+mech' +name+ pluton +'.png')
    plt.show()
