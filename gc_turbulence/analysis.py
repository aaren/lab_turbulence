"""Routines for analysing gravity current piv data"""

import numpy as np
import matplotlib.pyplot as plt

import modred as mr


class DMD(object):
    @staticmethod
    def calculate_dmd(data, n_modes=5):
        """Dynamic mode decomposition, using Uf as the series of vectors."""
        # create the matrix of snapshots by flattening the non
        # decomp axes so we have a 2d array where we index the
        # decomp axis like snapshots[:,i]
        # the decomposition axis is the x dimension of the front
        # relative data
        iz, ix, it = data.shape
        snapshots = data.transpose((0, 2, 1)).reshape((-1, ix))

        # remove nans
        # TODO: remove nans by interpolation earlier on
        # snapshots[np.where(np.isnan(snapshots))] = 0

        modes, ritz_values, norms \
            = mr.compute_DMD_matrices_snaps_method(snapshots, range(n_modes))

        # as array, reshape to data dims with mode number as first index
        reshaped_modes = modes.A.reshape((iz, it, -1)).transpose((2, 0, 1))
        return reshaped_modes

    @staticmethod
    def plot_dmd(modes, X, Z, T, data):
        # slice to get the coordinates out
        coords = np.s_[:, 0, :]

        fig, ax = plt.subplots(nrows=6, figsize=(12, 12))
        # plot decomp mean velocity
        mean = np.mean(data, axis=1)
        levels = np.linspace(-0.03, 0.04, 100)
        c0 = ax[0].contourf(T[coords], Z[coords], mean, levels=levels)

        ax[1].set_title('First mode of DMD')
        ax[1].set_xlabel('time after front passage')
        ax[1].set_ylabel('height')
        c1 = ax[1].contourf(T[coords], Z[coords], modes[0], levels=levels)

        ax[2].set_title('Second mode of DMD')
        ax[2].set_xlabel('time after front passage')
        ax[2].set_ylabel('height')
        # TODO: why does reshaped_modes seem to have a list of
        # duplicates?
        # Seems to be complex conjugates - why is this??
        c2 = ax[2].contourf(T[coords], Z[coords], modes[2], levels=c1.levels)

        ax[3].set_title('Third mode of DMD')
        ax[3].set_xlabel('time after front passage')
        ax[3].set_ylabel('height')
        c3 = ax[3].contourf(T[coords], Z[coords], modes[4], levels=c1.levels)

        ax[4].set_title('Fourth mode of DMD')
        ax[4].set_xlabel('time after front passage')
        ax[4].set_ylabel('height')
        c4 = ax[4].contourf(T[coords], Z[coords], modes[6], levels=c1.levels)

        ax[5].set_title('Fifth mode of DMD')
        ax[5].set_xlabel('time after front passage')
        ax[5].set_ylabel('height')
        c5 = ax[5].contourf(T[coords], Z[coords], modes[8], levels=c1.levels)

        fig.colorbar(c0, ax=ax[0], use_gridspec=True)
        fig.colorbar(c1, ax=ax[1], use_gridspec=True)
        fig.colorbar(c2, ax=ax[2], use_gridspec=True)
        fig.colorbar(c3, ax=ax[3], use_gridspec=True)
        fig.colorbar(c4, ax=ax[4], use_gridspec=True)
        fig.colorbar(c5, ax=ax[5], use_gridspec=True)

        fig.tight_layout()

        return fig


class Plotter(object):
    def __init__(self, run):
        """Plotting routines for turbulence analysis.

            run - ProcessedRun instance
        """
        self.run = run

    def mean_velocity(self, ax):
        u_mod_bar = self.mean_f(self.uf_abs)
        contourf = ax.contourf(u_mod_bar, self.levels_u)
        ax.set_title(r'Mean speed $\overline{|u|_t}(x, z)$')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def mean_velocity_Uf(self, ax):
        mean_Uf = self.mean_f(self.uf)
        contourf = ax.contourf(mean_Uf, self.levels_u)
        ax.set_title('Time averaged streamwise velocity')
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('height')
        return contourf

    def mean_velocity_Wf(self, ax):
        mean_Wf = self.mean_f(self.wf)
        contourf = ax.contourf(mean_Wf, self.levels_w)
        ax.set_title('Time averaged vertical velocity')
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('height')
        return contourf
