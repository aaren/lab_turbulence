#!/usr/bin/env python

"""
Interactive usage example:

    import plot
    r = plot.cache_test_run()
    # access the U array
    r.U

"""

import os
import argparse

import numpy as np
from scipy import stats
from scipy import ndimage as ndi
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# import modred as mr

import wavelets

from gc_turbulence import RawRun
from gc_turbulence import util


w_dir = '/home/eeaol/lab/data/flume2/'
plot_dir = os.path.join(os.environ['HOME'], 'code/gc_turbulence/demo/plots/')

# preliminary runs that we have so far
runs = ['3b4olxqo', '3ban2y82', '3bb5hsxk', '3bbn7639', '3bc4rhc0']

# limits of usability on the runs (first frame, last frame)
run_lims = {'3b4olxqo': (2200, 4400),
            '3ban2y82': (1700, 3200),
            '3bb5hsxk': (2400, 3600),
            '3bbn7639': (850, 2400),
            '3bc4rhc0': (1300, 3000),
            '3hxlfmtp': (0, -1),  # stereo run
            'r13_12_12c': (0, -1),}  # single camera of a stereo run


# plots to create by default (based on function names in PlotRun)
default_plots = ['hovmoller',
                 'mean_velocity',
                 'mean_velocity_Uf',
                 'mean_velocity_Wf',
                 'median_velocity',
                 'power',
                 'mean_vorticity',
                 'std_velocity',
                 'std_velocity_U',
                 'std_velocity_W',
                 'momentum_flux',
                 'mean_shear',
                 'wavelet', ]

# plots that are self contained (do their own axes composition)
special_plots = ['power',
                 'wavelet',
                 'hovmoller',
                 'autocorrelation',
                 'dmd']


def front_detect(U):
    # It would be useful to detect where the front is in each image, so
    # that we can do meaningful statistics relative to this.

    # how do we do this??

    # look at given x
    # only consider lowest half of z
    # average over this z range
    Z, X, T = U.shape
    # array of front locations in time, over all x
    z_mean = np.mean(U[:Z / 4, :, :], axis=0)
    smoothed_mean = ndi.gaussian_filter1d(z_mean, sigma=30, axis=1)
    # front position is given by the minimum of derivative of
    # velocity
    front_pos = np.nanargmin(np.diff(smoothed_mean, axis=1), axis=1)
    # FIXME: these are fake x values, n.b. != X
    front_pos_X = np.indices(front_pos.shape).squeeze()
    linear_fit = np.poly1d(np.polyfit(front_pos_X, front_pos, 1))
    # now given position x , get time of front pass t = linear_fit(x)
    return linear_fit


class PlotRun(object):
    def __init__(self, run, run_kwargs=None, t_width=800):
        self.index = run
        wd = os.path.join(w_dir, run)
        if not run_kwargs:
            run_kwargs = {'data_dir':  wd,
                          'index':     self.index,
                          'parallel':  True,
                          'caching':   True,
                          'limits':    run_lims[run]}
        self.r = RawRun(**run_kwargs)
        # hack, remove the x at the edges
        # TODO: this should be done with masking somewhere # previously
        for d in ('u', 'w', 't'):
            arr = getattr(self.r, d)
            setattr(self, d, arr[:, 15:-15, :])

        # time of front passage as f(x)
        self.tf = front_detect(self.U)

        # front velocity dx/dt
        # TODO: sort out non dimensional units
        self.front_velocity = 1 / self.tf[1]

        self.T_width = t_width
        self.front_offset = -10
        self.uf = self.reshape_to_current_relative(self.u)
        self.wf = self.reshape_to_current_relative(self.w)
        self.tf = self.reshape_to_current_relative(self.t)

        ## gradients
        self.duz, self.dux, self.dut = np.gradient(self.u)
        self.dwz, self.dwx, self.dwt = np.gradient(self.w)
        self.dufz, self.dufx, self.duft = np.gradient(self.uf)
        self.dwfz, self.dwfx, self.dwft = np.gradient(self.wf)

        ## reynolds decomposition, u = u_ + u'
        # mean_flow
        self.uf_ = self.mean_f(self.uf)
        self.wf_ = self.mean_f(self.wf)
        # perturbation (zero mean)
        self.ufp = self.uf - np.expand_dims(self.uf_, 1)
        self.wfp = self.wf - np.expand_dims(self.wf_, 1)

        ## derived properties
        # absolute velocity
        self.uf_abs = np.hypot(self.uf, self.wf)
        # vorticity
        self.vorticity = self.dwfx - self.dufz
        # vertical absolute velocity shear
        self.vertical_shear = np.hypot(self.dufz, self.dwfz)

        # ranges for plotting
        self.u_range = (-10, 5)
        self.w_range = (-2, 2)
        self.u_abs_range = (0,
                            np.hypot(np.abs(self.u_range).max(),
                                     np.abs(self.w_range).max()))

        # colour levels for contour plotting
        self.levels_u = np.linspace(*self.u_range, num=100)
        self.levels_w = np.linspace(*self.w_range, num=100)
        self.levels = self.levels_u

    @property
    def properties(self):
        """Define properties of quantities. Used in plotting to give a
        string to put in title, axes labels etc."""
        return {'u':  {'name': 'streamwise velocity',
                       'range': self.u_range},
                'uf': {'name': 'streamwise velocity',
                       'range': self.u_range},
                'w':  {'name': 'vertical velocity',
                       'range': self.w_range},
                'wf':  {'name': 'vertical velocity',
                        'range': self.w_range},
                'uf_': {'name': 'mean streamwise velocity',
                        'range': self.u_range},
                'wf_': {'name': 'mean vertical velocity',
                        'range': self.w_range},
                'uf_abs': {'name': 'absolute velocity',
                           'range': self.u_abs_range},
                'vorticity': {'name': 'vorticity',
                              'range': self.w_range},
                'vertical_shear': {'name': 'vertical_shear',
                                   'range': self.w_range}}

    def mean_f(self, x):
        """Compute mean over time varying axis of a front relative
        quantity, x.
        """
        # TODO: the axis used in nanmean is different for U and Uf
        # calcs - change Uf dims to make consistent?
        return stats.nanmean(x, axis=1)

    def median_f(self, x):
        """Compute median over time varying axis of a front relative
        quantity, x.
        """
        # TODO: the axis used in nanmean is different for U and Uf
        # calcs - change Uf dims to make consistent?
        return stats.nanmedian(x, axis=1)

    def rms_f(self, x):
        """Compute standard deviation over time varying axis of a
        front relative quantity, x.
        """
        # TODO: the axis used in nanmean is different for U and Uf
        # calcs - change Uf dims to make consistent?
        return stats.nanstd(x, axis=1)

    def reshape_to_current_relative(self, vel):
        """Take the velocity data and transform it to the current
        relative frame.
        """
        # tf is the time of front passage as f(x), i.e. supply this
        # with an argument in x and we get the corresponding time
        # reshape, taking a constant T time intervals behind front
        t0 = self.front_offset
        t1 = self.T_width
        tf = self.tf
        X = np.indices((vel.shape[1],)).squeeze()
        U_ = np.dstack(vel[:, x, int(tf(x)) + t0:int(tf(x)) + t1] for x in X)
        # reshape this to same dimensions as before
        Uf = np.transpose(U_, (0, 2, 1))
        # TODO: does axis 2 of Uf need to be reversed?
        # reverse axis so that time axis progresses as time in the
        # evolution of the front
        # Uf = Uf[:,:,::-1]
        return Uf

    def hovmoller(self, ax, quantity, zi=10):
        ax.set_xlabel('time')
        ax.set_ylabel('distance')
        contourf = ax.contourf(quantity[zi, :, :], levels=self.levels)
        return contourf

    def histogram(self, ax, quantity, range, bins=1000):
        uf = quantity.flatten()
        ax.hist(uf, bins=1000, range=range)
        title = 'Streamwise velocity distribution, run {run}'
        ax.set_title(title.format(run=self.index))
        ax.set_xlabel('Streamwise velocity, pixels')
        ax.set_ylabel('frequency')
        ax.set_yticks([])
        return

    def vertical_distribution(self, ax, quantity, range, bins=1000):
        """Make a contour plot of the vertical distribution of some
        quantity.

        Creates a histogram (over time and space) for each vertical
        level and then concatenates these together.
        """
        q = quantity
        # z is the first dimension
        # FIXME: this Z should come from self
        iZ = np.indices((q.shape[0],)).squeeze()

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

    def mean_velocity(self, ax):
        u_mod_bar = self.mean_f(self.uf_abs)
        contourf = ax.contourf(u_mod_bar, self.levels)
        ax.set_title(r'Mean speed $\overline{|u|_t}(x, z)$')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def mean_velocity_Uf(self, ax):
        mean_Uf = self.mean_f(self.uf)
        contourf = ax.contourf(mean_Uf, self.levels)
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

    def median_velocity(self, ax):
        median_u = self.median_f(self.uf_abs)
        contourf = ax.contourf(median_u, self.levels)
        ax.set_title('Median speed')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def std_velocity(self, ax):
        std_u = self.rms_f(self.uf_abs)
        contourf = ax.contourf(std_u, levels=np.linspace(0, 2, 100))
        ax.set_title('rms absolute velocity')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def std_velocity_U(self, ax):
        std_Uf = self.rms_f(self.uf)
        contourf = ax.contourf(std_Uf, levels=np.linspace(0, 2, 100))
        ax.set_title('rms streamwise velocity')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def std_velocity_W(self, ax):
        std_Wf = self.rms_f(self.wf)
        contourf = ax.contourf(std_Wf, levels=np.linspace(0, 2, 100))
        ax.set_title('rms vertical velocity')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def momentum_flux(self, ax):
        uw = self.ufp * self.wfp
        uw_ = np.abs(self.mean_f(uw))
        contourf = ax.contourf(uw_, 100)
        ax.set_title('momentum flux')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def power_spectrum(self, ax):
        times = self.tf
        # limit to bottom half of domain
        U = self.uf
        half = U.shape[1] / 2
        U = U[:, :half, :]
        # compute fft over time
        fft_U = np.fft.fft(U, axis=2)
        power_spectrum = np.abs(fft_U) ** 2
        freqs = np.fft.fftfreq(times.shape[-1], d=0.01)
        # power_spectrum is 3 dimensional. there is a 1d power
        # spectrum for each x, z point
        # we want to overlay all of these on the same plot
        # matplotlib can cope with a 2d array as y, with the first
        # dimension the same as that of the x axis
        # as the time dimension is the last one, we can flatten the
        # 3d array and resize the frequency array to match

        f_ps = power_spectrum.flatten()
        f_freqs = np.resize(freqs, f_ps.shape)

        # histogram the power data
        # http://stackoverflow.com/questions/10439961
        xlo, xhi = 0.1, 50
        ylo, yhi = 1E-5, 1E7
        res = 100
        X = np.logspace(np.log10(xlo), np.log10(xhi), res)
        Y = np.logspace(np.log10(ylo), np.log10(yhi), res)
        bins = (X, Y)
        thresh = 1
        hh, lx, ly = np.histogram2d(f_freqs, f_ps, bins=bins)

        # either mask out less than thresh, or set vmin=thresh in
        # pcolormesh
        mhh = np.ma.masked_where(hh < thresh, hh)

        pcolor = ax.pcolormesh(X, Y, mhh.T, cmap='jet',
                               norm=mpl.colors.LogNorm())

        # scatter plot for low density?
        # find what the points are
        # lhh = np.ma.masked_where(hh > thresh, hh)
        # low_f_freqs = f_freqs
        # ax_power.plot(low_f_freqs, low_f_ps, 'b.')

        # overlay a -5 / 3 line
        y0 = 1E3  # where to start in y? (power)
        x0, x1 = 3E-1, 1E1  # where in x should we go from and to? (power)
        x53 = np.linspace(x0, x1)
        y53 = x53 ** (-5 / 3) * y0
        ax.plot(x53, y53, 'k', linewidth=2)
        ax.text(x0, y0, '-5/3')

        ax.set_title('Power Spectrum')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')

        return pcolor

    def fft_U(self, ax):
        # limit to bottom half of domain
        ax.set_title('domain fft')
        U = self.uf
        half = U.shape[1] / 2
        U = U[:, :half, :]
        times = self.tf
        # compute fft over time
        fft_U = np.fft.fft(U, axis=2)
        freqs = np.fft.fftfreq(times.shape[-1], d=0.01)

        # compute average fft over domain
        domain_fft = stats.nanmean(stats.nanmean(fft_U))
        ax.plot(freqs, domain_fft.real, 'k.', label='Re')
        ax.plot(freqs, domain_fft.imag, 'r.', label='Im')
        ax.set_yscale('log')
        ax.set_ylim(1E-2, 1E3)
        ax.set_xlim(0, 50)
        ax.set_xscale('log')
        ax.legend()

        return

    def mean_vorticity(self, ax):
        mean_vorticity = self.mean_f(self.vorticity)
        contourf = ax.contourf(mean_vorticity, 100)
        ax.set_title('Mean vorticity')
        return contourf

    def mean_shear(self, ax):
        mean_shear = self.mean_f(self.vertical_shear)
        contourf = ax.contourf(mean_shear, 100)
        ax.set_title('Mean vertical shear')
        return contourf

    def wavelet(self, ax):
        ax.set_title('Wavelet power spectrum (morlet)')
        ax.set_xlabel('time after front passage (s)')
        ax.set_ylabel('equivalent fourier frequency (Hz)')

        sig = self.u[20, 20, :]
        wa = wavelets.WaveletAnalysis(sig, dt=0.01, wavelet=wavelets.Morlet(),
                                      unbias=True)

        fourier_freqs = 1 / wa.fourier_periods

        T, S = np.meshgrid(wa.time, fourier_freqs)

        contourf = ax.contourf(T, S, wa.wavelet_power, 100)

        # shade the region between the edge and coi
        C, S = wa.coi
        # fourier period
        Fp = wa.fourier_period(S)
        # fourier freqs
        Ff = 1 / Fp
        ff_min = fourier_freqs.min()
        ax.fill_between(x=C, y1=Ff, y2=ff_min, color='gray', alpha=0.3)

        ax.set_yscale('log')

        return contourf

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

    def plot_hovmoller(self, zi=10):
        """Create a hovmoller of the streamwise velocity at the
        given z index (default 10) and overlay the detected front
        position vector.

        Also plot the front relative hovmoller
        """
        fig, axes = plt.subplots(nrows=3)
        ax_avg, ax_U, ax_Uf = axes

        self.mean_velocity_Uf(ax_avg)
        ax_avg.axhline(zi, linewidth=2, color='black')

        ax_U.set_title('Hovmoller of streamwise velocity')
        hovmoller_U = self.hovmoller(ax_U, quantity=self.u)
        # over plot line of detected front passage
        # FIXME: this x should be from self
        x = np.indices((self.u.shape[1],)).squeeze()
        tf = self.tf
        ax_U.plot(tf(x), x, label='detected front')

        ax_U.set_title('Hovmoller of shifted streamwise velocity')
        self.hovmoller(ax_Uf, quantity=self.uf)
        # over plot line of detected front passage
        ax_Uf.axvline(-self.front_offset, label='detected front')

        ax_U.legend()
        ax_Uf.legend()

        # make space for shared colorbar
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(hovmoller_U, cax=cax, use_gridspec=True)

        return fig

    def plot_distribution(self, q, q_range, name):
        fig, axes = plt.subplots(nrows=2)
        ax_all_domain_dist, ax_vertical_dist = axes

        self.histogram(ax_all_domain_dist, q, range=q_range)
        self.vertical_distribution(ax_vertical_dist, q, range=q_range)

        all_domain_title = '{quantity} distribution, run {run}'
        all_domain_title = all_domain_title.format(run=self.index,
                                                   quantity=name)
        all_domain_xlabel = '{quantity}'.format(quantity=name)
        ax_all_domain_dist.set_title(all_domain_title)
        ax_all_domain_dist.set_xlabel(all_domain_xlabel)

        vertical_dist_title = '{quantity} distribution'.format(quantity=name)
        vertical_dist_xlabel = '{quantity}'.format(quantity=name)
        ax_vertical_dist.set_title(vertical_dist_title)
        ax_vertical_dist.set_xlabel(vertical_dist_xlabel)

        fig.tight_layout()

        return fig

    def plot_autocorrelation(self):
        """Plot the autocorrelation as a function of height of
        the mean front relative frame.
        """
        fig, ax = plt.subplots()
        U = stats.nanmean(self.uf, axis=1)
        # correlate two 1d arrays
        # np.correlate(U, U, mode='full')[len(U) - 1:]
        # but we want to autocorrelate a 2d array over a given
        # axis
        N = U.shape[1]
        pad_N = N * 2 - 1
        s = np.fft.fft(U, n=pad_N, axis=1)
        acf = np.real(np.fft.ifft(s * s.conjugate(), axis=1))[:, :N]
        # normalisation
        acf0 = np.expand_dims(acf[:, 0], 1)
        acf = acf / acf0

        fig, ax = plt.subplots(nrows=2)
        c0 = ax[0].contourf(U, self.levels)
        c1 = ax[1].contourf(acf, 100)

        fig.colorbar(c0, ax=ax[0], use_gridspec=True)
        fig.colorbar(c1, ax=ax[1], use_gridspec=True)

        ax[0].set_title(r'$\overline{u_x}(z, t)$')
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('z')

        ax[1].set_title('autocorrelation')
        ax[1].set_xlabel('lag')
        ax[1].set_ylabel('z')

        fig.tight_layout()

        return fig

    def plot_power(self):
        # fourier transform over the time axis
        fig, axes = plt.subplots(nrows=3)
        ax_location, ax_power, ax_fft = axes

        self.mean_velocity_Uf(ax_location)
        self.power_spectrum(ax_power)
        self.fft_U(ax_fft)

        fig.tight_layout()
        return fig

    def plot_wavelet(self):
        fig, ax = plt.subplots()
        self.wavelet(ax)
        return fig

    def plot_time_slices(self):
        """If we take a vertical profile at a given horizontal
        position we can make a contour plot of some quantity over
        the vertical axis and time.

        Similarly, for a given moment in time we could make a contour
        plot over the horizontal and vertical axes.

        If we make a plots over all of the possible vertical profiles,
        we end up with a series of plots that can be animated.
        """
        U = self.r.u[:, 15:-15, :]
        T = range(U.shape[2])
        kwarglist = [dict(t=t,
                          index=self.index,
                          U=U,
                          levels=self.levels,
                          fname=self.time_slice_path(t))
                     for t in T]
        util.parallel_process(plot_time_slice, kwarglist=kwarglist)

    def plot_vertical_transects(self):
        """If we take a vertical profile at a given horizontal
        position we can make a contour plot of some quantity over
        the vertical axis and time.

        Similarly, for a given moment in time we could make a contour
        plot over the horizontal and vertical axes.

        If we make a plots over all of the possible vertical profiles,
        we end up with a series of plots that can be animated.
        """
        U = self.uf
        # TODO: why is this X reversed? should get this from self?
        X = range(U.shape[1])[::-1]
        kwarglist = [dict(x=x,
                          index=self.index,
                          U=U,
                          levels=self.levels,
                          fname=self.vertical_transect_path(x))
                     for x in X]
        util.parallel_process(plot_vertical_transect, kwarglist=kwarglist)

    def time_slice_path(self, t):
        fname = 'time_slices/time_slice_{r}_{t:0>4d}.png'
        fpath = os.path.join(plot_dir, fname.format(r=self.index, t=t))
        return fpath

    def vertical_transect_path(self, x):
        fname = 'vertical_transects/vertical_transect_{r}_{x:0>4d}.png'
        fpath = os.path.join(plot_dir, fname.format(r=self.index, x=x))
        return fpath

    def plot_dmd(self):
        """Dynamic mode decomposition, using Uf as the series of vectors."""
        n_modes = 10
        U = self.uf
        # put the decomposition axis last
        UT = U.transpose(0, 2, 1)
        # create the matrix of snapshots by flattening the non
        # decomp axes so we have a 2d array where we index the
        # decomp axis like snapshots[:,i]
        snapshots = UT.reshape((-1, UT.shape[-1]))

        # remove nans
        # TODO: remove nans by interpolation earlier on
        snapshots[np.where(np.isnan(snapshots))] = 0

        modes, ritz_values, norms \
            = mr.compute_DMD_matrices_snaps_method(snapshots, range(n_modes))

        # as array, reshape to data dims
        reshaped_modes = modes.A.T.reshape((-1,) + UT.shape[:-1])

        fig, ax = plt.subplots(nrows=3)
        c0 = self.mean_velocity_Uf(ax[0])

        ax[1].set_title('First mode of DMD')
        ax[1].set_xlabel('time after front passage')
        ax[1].set_ylabel('height')
        c1 = ax[1].contourf(reshaped_modes[0], 100)

        ax[2].set_title('Second mode of DMD')
        ax[2].set_xlabel('time after front passage')
        ax[2].set_ylabel('height')
        # TODO: why does reshaped_modes seem to have a list of
        # duplicates?
        # Seems to be complex conjugates - why is this??
        c2 = ax[2].contourf(reshaped_modes[2], 100, levels=c1.levels)

        fig.colorbar(c0, ax=ax[0], use_gridspec=True)
        fig.colorbar(c1, ax=ax[1], use_gridspec=True)
        fig.colorbar(c2, ax=ax[2], use_gridspec=True)

        fig.tight_layout()

        return fig

    def plot_figure(self, quantity, colorbar=True, quiver=True):
        fig, ax = plt.subplots()
        if quantity in special_plots:
            fig = getattr(self, 'plot_' + quantity)()
        else:
            plot_func = getattr(self, quantity)(ax)
            if colorbar:
                fig.colorbar(plot_func)
            if quiver:
                self.overlay_velocities(ax)
        return fig

    def main(self, args):
        """args is an Argparse namespace with command line options.

        plots is a list of plotting functions to execute and save.

        funcs is a list of plotting functions to execute. This is used
        for multiprocessing plotting functions that don't return a single
        figure.
        """
        for plot in args.plots:
            if plot == 'no_plot':
                break
            print "plotting", plot

            fig = self.plot_figure(plot)

            fformat = '{plot}_{index}.{ext}'
            fname = fformat.format(plot=plot, index=self.index, ext='png')
            fpath = os.path.join(plot_dir, fname)
            fig.savefig(fpath)

        if args.distributions == 'all':
            distributions = ['Uf', 'Wf', 'uf_abs',
                             'vorticity', 'vertical_shear']
        else:
            distributions = args.distributions
        for dist in distributions:
            range = self.properties[dist]['range']
            name = self.properties[dist]['name']
            print "plotting distribution", dist, name
            fig = self.plot_distribution(getattr(self, dist), range, name)

            fformat = 'distribution_{q}_{index}.{ext}'
            fname = fformat.format(q=dist, index=self.index, ext='png')
            fpath = os.path.join(plot_dir, fname)
            fig.savefig(fpath)

        if args.funcs:
            for func in args.funcs:
                print "multiprocessing", func
                f = getattr(self, 'plot_' + func)
                f()


@util.parallel_stub
def plot_time_slice(index, t, U, fname, levels):
    """Plot a single vertical transect. Function external to class to
    allow multiprocessing.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = 'Time slice {r} {t:0>4d}'.format(r=index, t=t)
    ax.set_title(title)
    U = U[:, :, t]
    contourf = ax.contourf(U, levels)
    fig.colorbar(contourf)
    util.makedirs_p(os.path.dirname(fname))
    fig.savefig(fname)


@util.parallel_stub
def plot_vertical_transect(index, x, U, fname, levels):
    """Plot a single vertical transect. Function external to class to
    allow multiprocessing.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = 'Vertical transect {r} {x:0>4d}'.format(r=index, x=x)
    ax.set_title(title)
    U = U[:, x, :]
    contourf = ax.contourf(U, levels)
    fig.colorbar(contourf)
    util.makedirs_p(os.path.dirname(fname))
    fig.savefig(fname)


def test_run(index='3ban2y82', reload=False):
    """Return a test run"""
    wd = os.path.join(w_dir, index)
    start = run_lims[index][0]
    end = run_lims[index][1]
    run_kwargs = {'data_dir':     wd,
                  'index':        index,
                  'parallel':     True,
                  'caching':      True,
                  'cache_reload': reload,
                  'limits':       (start, end)}
    r = PlotRun(run=index, run_kwargs=run_kwargs, t_width=400)
    return r


def cache_test_run(index='3ban2y82'):
    """Return a test run"""
    wd = '/home/aaron/code/gc_turbulence/tests/ex_data/tmp'
    # TODO: make this path more explicit w/o being machine specific
    start = run_lims[index][0]
    end = run_lims[index][1]
    run_kwargs = {'data_dir':     wd,
                  'index':        index,
                  'parallel':     True,
                  'caching':      True,
                  'cache_reload': False,
                  'limits':       (start, end)}
    r = PlotRun(run=index, run_kwargs=run_kwargs, t_width=800)
    return r


if __name__ == '__main__':
    test_run_index = '3ban2y82'
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        help="single run test mode",
                        nargs='?',
                        const=test_run_index)
    parser.add_argument("--cache_test",
                        help="single run test mode, load from cache",
                        nargs='?',
                        const=test_run_index)
    plot_help = ("List of plots to make. Valid names are any of "
                 "{plots}. Default is to plot everything. Use 'no_plot' "
                 "to plot nothing.")
    parser.add_argument('plots',
                        help=plot_help.format(plots=default_plots),
                        nargs='*',
                        type=str,
                        default=default_plots)
    parser.add_argument("--run",
                        help="specify runs to process",
                        nargs='*',
                        dest='run',
                        default=runs)
    parser.add_argument("--distributions",
                        help="specify distribution plots to make",
                        nargs='?',
                        default=[],
                        const='all')
    parser.add_argument("--vertical_transects",
                        help="compute vertical transects (multiprocessing)",
                        action='append_const',
                        const='vertical_transects',
                        dest='funcs')
    parser.add_argument("--time_slices",
                        help="compute time slices (multiprocessing)",
                        action='append_const',
                        const='time_slices',
                        dest='funcs')
    parser.add_argument("--dmd",
                        help="dynamic mode decomposition",
                        action='append_const',
                        const='dmd',
                        dest='funcs')
    # TODO: add argument for reload without plotting anything
    parser.add_argument("--reload",
                        help="force reloading cache, n.b. deletes "
                             "old cache file. To reload cache without "
                             "plotting anything, give 'no_plot' as plot "
                             "argument",
                        action='store_true')
    parser.add_argument("--stereo",
                        help="specify stereo run",
                        action='store_true')
    args = parser.parse_args()

    if not ('HOSTNAME' in os.environ) \
       or (os.environ['HOSTNAME'] != 'doug-and-duck'):
        print "Not running on doug-and-duck, are you sure?"
        raw_input()

    if args.test:
        r = test_run(index=test_run_index, reload=args.reload)
        r.main(args)

    elif args.cache_test:
        r = cache_test_run(index=test_run_index)
        r.main(args)

    else:
        print "Processing runs: ", args.run
        for run in args.run:
            print "Extracting " + run + "...\n"
            wd = os.path.join(w_dir, run)
            run_kwargs = {'data_dir':     wd,
                          'index':        run,
                          'parallel':     True,
                          'caching':      True,
                          'stereo':       args.stereo,
                          'cache_reload': args.reload,
                          'limits':       run_lims[run]}
            pr = PlotRun(run, run_kwargs)
            pr.main(args)
