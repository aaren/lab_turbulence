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
from scipy import signal
from scipy import ndimage as ndi
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.environ['HOME'], 'code'))
from gc_turbulence.gc_turbulence.turbulence import SingleLayerRun
import gc_turbulence.gc_turbulence.util as util


w_dir = '/home/eeaol/lab/data/flume2/'
plot_dir = os.path.join(os.environ['HOME'], 'code/gc_turbulence/demo/plots/')

# preliminary runs that we have so far
runs = ['3b4olxqo', '3ban2y82', '3bb5hsxk', '3bbn7639', '3bc4rhc0']

# limits of usability on the runs (first frame, last frame)
run_lims = {'3b4olxqo': (2200, 4400),
            '3ban2y82': (1700, 3200),
            '3bb5hsxk': (2400, 3600),
            '3bbn7639': (850, 2400),
            '3bc4rhc0': (1300, 3000)}


# plots to create by default (based on function names in PlotRun)
default_plots = ['hovmoller',
                 'histogram_U',
                 'histogram_W',
                 'mean_velocity',
                 'mean_velocity_Uf',
                 'mean_velocity_Wf',
                 'median_velocity',
                 'power',
                 'mean_vorticity',
                 'std_velocity',
                 'std_velocity_U',
                 'std_velocity_W',
                 'mean_shear',
                 'wavelet',
                 'autocorrelation']


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
        self.r = SingleLayerRun(**run_kwargs)
        self.u_range = (-10, 5)
        self.w_range = (-2, 2)
        # hack, remove the x at the edges
        for d in ('U', 'W', 'T'):
            arr = getattr(self.r, d)
            setattr(self, d, arr[:, 15:-15, :])

        # time of front passage as f(x)
        self.tf = front_detect(self.U)

        # front velocity dx/dt
        # TODO: sort out non dimensional units
        self.front_velocity = 1 / self.tf[1]

        self.T_width = t_width
        self.front_offset = -50
        self.Uf = self.reshape_to_current_relative(self.U, self.front_offset, self.T_width)
        self.Wf = self.reshape_to_current_relative(self.W, self.front_offset, self.T_width)
        self.Tf = self.reshape_to_current_relative(self.T, self.front_offset, self.T_width)

        ## gradients
        self.dUz, self.dUx, self.dUt = np.gradient(self.U)
        self.dWz, self.dWx, self.dWt = np.gradient(self.W)
        self.dUfz, self.dUfx, self.dUft = np.gradient(self.Uf)
        self.dWfz, self.dWfx, self.dWft = np.gradient(self.Wf)

        ## derived properties
        # absolute velocity
        self.uf_abs = np.hypot(self.Uf, self.Wf)
        # vorticity
        self.vorticity = self.dWfx - self.dUfz
        # vertical absolute velocity shear
        self.vertical_shear = np.hypot(self.dUfz, self.dWfz)

        # colour levels for contour plotting
        self.levels_u = np.linspace(*self.u_range, num=100)
        self.levels_w = np.linspace(*self.w_range, num=100)
        self.levels = self.levels_u

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

    def reshape_to_current_relative(self, vel, T0, T1):
        """Take the velocity data and transform it to the current
        relative frame.
        """
        # tf is the time of front passage as f(x), i.e. supply this
        # with an argument in x and we get the corresponding time
        # reshape, taking a constant T time intervals behind front
        tf = self.tf
        X = np.indices((vel.shape[1],)).squeeze()
        U_ = np.dstack(vel[:, x, int(tf(x)) + T0:int(tf(x)) + T1] for x in X)
        # reshape this to same dimensions as before
        Uf = np.transpose(U_, (0, 2, 1))
        return Uf

    def hovmoller(self, ax, quantity, zi=10):
        ax.set_xlabel('time')
        ax.set_ylabel('distance')
        contourf = ax.contourf(quantity[zi, :, :], levels=self.levels)
        return contourf

    def histogram(self, ax, quantity):
        uf = quantity.flatten()
        ax.hist(uf, bins=1000, range=self.u_range)
        title = 'Streamwise velocity distribution, run {run}'
        ax.set_title(title.format(run=self.index))
        ax.set_xlabel('Streamwise velocity, pixels')
        ax.set_ylabel('frequency')
        ax.set_yticks([])
        return

    def vertical_distribution(self, ax, quantity):
        # FIXME: this Z should come from self
        q = quantity
        Z = np.indices((q.shape[0],)).squeeze()
        U_bins = np.histogram(q[0], range=self.u_range, bins=1000)[1]
        histograms = (np.histogram(q[z], range=self.u_range, bins=1000)[0]
                      for z in xrange(q.shape[0]))
        hist_z = np.dstack(histograms).squeeze()

        levels = np.linspace(0, 500, 100)
        contourf = ax.contourf(U_bins[1:], Z, hist_z.T, levels=levels)

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
        mean_Uf = self.mean_f(self.Uf)
        contourf = ax.contourf(mean_Uf, self.levels)
        ax.set_title('Time averaged streamwise velocity')
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('height')
        return contourf

    def mean_velocity_Wf(self, ax):
        mean_Wf = self.mean_f(self.Wf)
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
        std_Uf = self.rms_f(self.Uf)
        contourf = ax.contourf(std_Uf, levels=np.linspace(0, 2, 100))
        ax.set_title('rms streamwise velocity')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def std_velocity_W(self, ax):
        std_Wf = self.rms_f(self.Wf)
        contourf = ax.contourf(std_Wf, levels=np.linspace(0, 2, 100))
        ax.set_title('rms vertical velocity')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        return contourf

    def power_spectrum(self, ax):
        times = self.Tf
        # limit to bottom half of domain
        U = self.Uf
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
        # http://stackoverflow.com/questions/10439961/efficiently-create-a-density-plot-for-high-density-regions-points-for-sparse-re
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
        U = self.Uf
        half = U.shape[1] / 2
        U = U[:, :half, :]
        times = self.Tf
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
        ax.set_title('Wavelet analysis (ricker)')
        ax.set_xlabel('time')
        ax.set_ylabel('period')

        wavelet_function = signal.wavelets.ricker

        scales = np.arange(1, 1500)

        wt = signal.cwt(self.U[20, 20, :], wavelet_function, scales)
        contourf = ax.contourf(wt, 100)
        return contourf

    def plot_figure(self, quantity, colorbar=True):
        fig, ax = plt.subplots()
        special_plots = ['power', 'wavelet', 'hovmoller', 'histogram_U',
                         'histogram_W', 'autocorrelation']
        if quantity in special_plots:
            fig = getattr(self, 'plot_' + quantity)()
        else:
            plot_func = getattr(self, quantity)(ax)
            if colorbar:
                fig.colorbar(plot_func)
        return fig

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
        hovmoller_U = self.hovmoller(ax_U, quantity=self.U)
        # over plot line of detected front passage
        # FIXME: this x should be from self
        x = np.indices((self.U.shape[1],)).squeeze()
        tf = self.tf
        ax_U.plot(tf(x), x, label='detected front')

        ax_U.set_title('Hovmoller of shifted streamwise velocity')
        self.hovmoller(ax_Uf, quantity=self.Uf)
        # over plot line of detected front passage
        ax_Uf.axvline(-self.front_offset, label='detected front')

        ax_U.legend()
        ax_Uf.legend()

        # make space for shared colorbar
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(hovmoller_U, cax=cax, use_gridspec=True)

        return fig

    def plot_histogram_U(self):
        fig, axes = plt.subplots(nrows=2)
        ax_all_domain_dist, ax_vertical_dist = axes

        self.histogram(ax_all_domain_dist, self.U)
        self.vertical_distribution(ax_vertical_dist, self.U)

        title = 'Streamwise velocity distribution, run {run}'
        ax_all_domain_dist.set_title(title.format(run=self.index))
        ax_all_domain_dist.set_xlabel('Streamwise velocity, pixels')

        ax_vertical_dist.set_title('streamwise velocity distribution')
        ax_vertical_dist.set_xlabel('streamwise velocity')

        fig.tight_layout()

        return fig

    def plot_histogram_W(self):
        fig, axes = plt.subplots(nrows=2)
        ax_all_domain_dist, ax_vertical_dist = axes

        self.histogram(ax_all_domain_dist, self.W)
        self.vertical_distribution(ax_vertical_dist, self.W)

        title = 'Vertical velocity distribution, run {run}'
        ax_all_domain_dist.set_title(title.format(run=self.index))
        ax_all_domain_dist.set_xlabel('Vertical velocity, pixels')

        ax_vertical_dist.set_title('vertical velocity distribution')
        ax_vertical_dist.set_xlabel('vertical velocity')

        fig.tight_layout()

        return fig

    def plot_autocorrelation(self):
        """Plot the autocorrelation as a function of height of
        the mean front relative frame.
        """
        fig, ax = plt.subplots()
        U = stats.nanmean(self.Uf, axis=1)
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
        U = self.r.U[:, 15:-15, :]
        T = range(U.shape[2])
        arglist = [dict(t=t,
                        index=self.index,
                        U=U,
                        levels=self.levels,
                        fname=self.time_slice_path(t))
                   for t in T]
        util.parallel_process(plot_time_slice, arglist)

    def plot_vertical_transects(self):
        """If we take a vertical profile at a given horizontal
        position we can make a contour plot of some quantity over
        the vertical axis and time.

        Similarly, for a given moment in time we could make a contour
        plot over the horizontal and vertical axes.

        If we make a plots over all of the possible vertical profiles,
        we end up with a series of plots that can be animated.
        """
        U = self.Uf
        X = range(U.shape[1])[::-1]
        arglist = [dict(x=x,
                        index=self.index,
                        U=U,
                        levels=self.levels,
                        fname=self.vertical_transect_path(x))
                   for x in X]
        util.parallel_process(plot_vertical_transect, arglist)

    def time_slice_path(self, t):
        fname = 'time_slices/time_slice_{r}_{t:0>4d}.png'
        fpath = os.path.join(plot_dir, fname.format(r=self.index, t=t))
        return fpath

    def vertical_transect_path(self, x):
        fname = 'vertical_transects/vertical_transect_{r}_{x:0>4d}.png'
        fpath = os.path.join(plot_dir, fname.format(r=self.index, x=x))
        return fpath

    def main(self, plots=default_plots, funcs=None):
        """plots is a list of plotting functions to execute and save.

        funcs is a list of plotting functions to execute. This is used
        for multiprocessing plotting functions that don't return a single
        figure.
        """
        for plot in plots:
            if plot == 'no_plot':
                break
            print "plotting", plot

            fig = self.plot_figure(plot)

            fformat = '{plot}_{index}.{ext}'
            fname = fformat.format(plot=plot, index=self.index, ext='png')
            fpath = os.path.join(plot_dir, fname)
            fig.savefig(fpath)

        if funcs:
            for func in funcs:
                print "multiprocessing", func
                f = getattr(self, 'plot_' + func)
                f()


def plot_time_slice(args):
    """Plot a single vertical transect. Function external to class to
    allow multiprocessing.
    """
    index = args["index"]
    t = args["t"]
    U = args["U"]
    pbar = args['pbar']
    fname = args['fname']
    queue = args['queue']
    levels = args['levels']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = 'Time slice {r} {t:0>4d}'.format(r=index, t=t)
    ax.set_title(title)
    U = U[:, :, t]
    contourf = ax.contourf(U, levels)
    fig.colorbar(contourf)
    util.makedirs_p(os.path.dirname(fname))
    fig.savefig(fname)
    queue.put(1)
    pbar.update()


def plot_vertical_transect(args):
    """Plot a single vertical transect. Function external to class to
    allow multiprocessing.
    """
    index = args["index"]
    x = args["x"]
    U = args["U"]
    pbar = args['pbar']
    fname = args['fname']
    queue = args['queue']
    levels = args['levels']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = 'Vertical transect {r} {x:0>4d}'.format(r=index, x=x)
    ax.set_title(title)
    U = U[:, x, :]
    contourf = ax.contourf(U, levels)
    fig.colorbar(contourf)
    util.makedirs_p(os.path.dirname(fname))
    fig.savefig(fname)
    queue.put(1)
    pbar.update()


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
    # TODO: add argument for reload without plotting anything
    parser.add_argument("--reload",
                        help="force reloading cache, n.b. deletes "
                             "old cache file. To reload cache without "
                             "plotting anything, give 'no_plot' as plot "
                             "argument",
                        action='store_true')
    args = parser.parse_args()

    if not ('HOSTNAME' in os.environ) or (os.environ['HOSTNAME'] != 'doug-and-duck'):
        print "Not running on doug-and-duck, are you sure?"
        raw_input()

    if args.test:
        r = test_run(index=test_run_index, reload=args.reload)
        r.main(plots=args.plots, funcs=args.funcs)

    elif args.cache_test:
        r = cache_test_run(index=test_run_index)
        r.main(plots=args.plots, funcs=args.funcs)

    else:
        for run in runs:
            print "Extracting " + run + "...\n"
            wd = os.path.join(w_dir, run)
            run_kwargs = {'data_dir':     wd,
                          'index':        run,
                          'parallel':     True,
                          'caching':      True,
                          'cache_reload': args.reload,
                          'limits':       run_lims[run]}
            pr = PlotRun(run, run_kwargs)
            pr.main(plots=args.plots, funcs=args.funcs)
