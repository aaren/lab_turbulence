"""Routines for analysing gravity current piv data.

Each class constitutes a different sort of analysis
and has methods for producing plots of that analysis.
Ideally these will take an axes instance as an argument
so that we can compose plots with multiple analyses as
subplots.

Usage:

    import gc_turbulence as g

    run = g.ProcessedRun(cache_path)

    dist = g.analysis.Distributions(run)

    fig, ax = plt.subplots()
    dist.plot_histogram(ax)

or perhaps compose these into ProcessedRun:

    run = g.ProcessedRun(cache_path)
    run.distributions.plot_zt_histogram(ax)

Alternately, we can set this up with a collection of classes that
have largely static methods, which we then feed data to.

The advantage of this is that we aren't strongly coupled to a run
and can analyse e.g. ensembles easier.

The disadvantage is that we aren't strongly coupled to a run and
we might as well just use the libraries directly.

Why would we even use classes??

Because some of the analysis is computationally expensive and the
variables computed are used more than once. It makes sense to save
them as shared state.
"""
import functools

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from sklearn.neighbors import KernelDensity

from .attributes import AnalysisAttributes
from .runbase import H5Cache


def subplot(plot_function):
    """Wrapper for functions that plot on a matplotlib axes instance
    that autocreates a figure and axes instance if ax is not
    supplied.
    """
    @functools.wraps(plot_function)
    def f(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            plot_function(self, ax, **kwargs)
            return fig
        else:
            return plot_function(self, ax, **kwargs)
    return f


class AnalysisBase(object):
    def __init__(self, run, **kwargs):
        self.run = run
        for k, v in kwargs.items():
            setattr(self, k, v)


class Basic(AnalysisBase):
    def __init__(self, run):
        """Plotting routines for turbulence analysis.

            run - ProcessedRun instance
        """
        self.run = run

        fr = np.s_[:, 0, :]
        self.tf = run.Tf[fr]
        self.zf = run.Zf[fr]

        self.tf_ = run.Tf_[fr]
        self.zf_ = run.Zf_[fr]

        self.uf = run.Uf
        self.uf_ = run.Uf_

        self.levels_u = 100
        self.levels_u_ = np.linspace(-0.5, 0.5, 100)

        # derived properties
        # absolute velocity
        self.uf_abs = np.hypot(run.Uf, run.Wf)
        self.uf__abs = np.hypot(run.Uf_, run.Wf_)

    def mean_f(self, x):
        """Compute mean over time varying axis of a front relative
        quantity, x.
        """
        # TODO: the axis used in nanmean is different for U and Uf
        # calcs - change Uf dims to make consistent?
        return np.mean(x, axis=1)

    @subplot
    def mean_velocity(self, ax):
        """Take an axes instance and contour plot the mean speed
        in front relative coordinates.
        """
        u_mod_bar = self.mean_f(self.uf_abs)
        contourf = ax.contourf(u_mod_bar, self.levels_u)
        ax.set_title(r'Mean speed $\overline{|u|_t}(x, z)$')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    @subplot
    def mean_velocity_(self, ax):
        """Take an axes instance and contour plot the mean speed
        in front relative coordinates.
        """
        u_mod_bar = self.mean_f(self.uf__abs)
        contourf = ax.contourf(u_mod_bar, self.levels_u_)
        ax.set_title(r'Mean speed $\overline{|u|_t}(x, z)$')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    @subplot
    def mean_velocity_Uf(self, ax):
        mean_Uf = self.mean_f(self.uf)
        contourf = ax.contourf(self.tf, self.zf, mean_Uf, self.levels_u)
        ax.set_title('Time averaged streamwise velocity')
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('height')
        return contourf

    @subplot
    def mean_velocity_Wf(self, ax):
        mean_Wf = self.mean_f(self.wf)
        contourf = ax.contourf(mean_Wf, self.levels_w)
        ax.set_title('Time averaged vertical velocity')
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('height')
        return contourf

    @subplot
    def overlay_velocities(self, ax):
        """Given an axes instance, overlay a quiver plot
        of Uf_ and Wf_.

        Uses interpolation (scipy.ndimage.zoom) to reduce
        number of quivers to readable number.

        Will only work sensibly if the thing plotted in ax
        has same shape as Uf_
        """
        zoom_factor = (0.5, 0.05)
        # TODO: proper x, z
        Z, X = np.indices(self.uf_.shape)

        # TODO: are the velocities going at the middle of their grid?
        # NB. these are not averages. ndi.zoom makes a spline and
        # then interpolates a value from this
        # TODO: gaussian filter first?
        # both are valid approaches
        Xr = ndi.zoom(X, zoom_factor)
        Zr = ndi.zoom(Z, zoom_factor)
        Uf_r = ndi.zoom(self.uf_, zoom_factor)
        Wf_r = ndi.zoom(self.wf_, zoom_factor)

        ax.quiver(Xr, Zr, Uf_r, Wf_r, scale=100)


class DMD(AnalysisBase):
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


class Histograms():
    @staticmethod
    def plot_time_histogram(ax, data, bins, where=np.s_[:], **kwargs):
        """Plot a histogram of a quantity through time.

        bins - edges of the histogram bins
        where - z index or slice object to use"""
        H, edges = np.histogramdd(data, bins=bins, normed=True)

        # hide empty bins
        Hmasked = np.ma.masked_where(H == 0, H)
        xedges, yedges = edges[:2]
        if 'levels' not in kwargs:
            kwargs['levels'] = np.linspace(0, 10)

        ax.contourf(xedges[1:], yedges[1:], Hmasked.T[where], **kwargs)
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('vertical velocity')
        return ax

    def vertical_histogram(self, ax, quantity, bins, levels):
        """Make a contour plot of the vertical distribution of some
        quantity.

        Creates a histogram (over time and space) for each vertical
        level and then concatenates these together.
        """
        iZ = None  # FIXME

        q = quantity
        hist_kwargs = {'range': range, 'bins': bins}
        q_bins = np.histogram(q[0], **hist_kwargs)[1]

        hist_z = np.vstack(np.histogram(q[z], **hist_kwargs)[0] for z in iZ)

        # TODO: determine from histogram?
        levels = np.linspace(0, 500, 100)
        # TODO: change iZ -> dimensioned Z
        # TODO: q_bins[1:] is a hack
        contourf = ax.contourf(q_bins[1:], iZ, hist_z, levels=levels)

        ax.set_title('distribution function')
        ax.set_xlabel('quantity')
        ax.set_ylabel('z')
        return contourf


class KDE(object):
    @staticmethod
    def fit_kde(data, coords, bandwidth, rtol=1E-3):
        kde = KernelDensity(bandwidth=bandwidth, rtol=rtol)
        kde.fit(data.T)
        log_pdf = kde.score_samples(coords.T)
        return np.exp(log_pdf)


class AnalysisRun(AnalysisAttributes, H5Cache):
    def __init__(self, cache_path=None, load=True):
        """A run ready for analysis.

        cache_path - hdf5 to load from
        load - whether to load hdf5
        """
        self.init(cache_path=cache_path, load=load)
        if not self.cache_path:
            return

        self.index = self.attributes['run_index']
        self.has_executed = False
