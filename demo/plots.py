import os
import argparse

import numpy as np
from scipy import stats
from scipy import ndimage as ndi
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from gc_turbulence.gc_turbulence.turbulence import SingleLayer2dRun
import gc_turbulence.gc_turbulence.util as util


w_dir = '/home/eeaol/lab/data/flume2/'
plot_dir = '/home/eeaol/code/gc_turbulence/demo/plots/'

# preliminary runs that we have so far
runs = ['3b4olxqo', '3ban2y82', '3bb5hsxk', '3bbn7639', '3bc4rhc0']

# limits of usability on the runs (first frame, last frame)
run_lims = {'3b4olxqo': (2200, 4400),
            '3ban2y82': (1700, 3200),
            '3bb5hsxk': (2400, 3600),
            '3bbn7639': (850, 2400),
            '3bc4rhc0': (1300, 3000)}


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
        self.r = SingleLayer2dRun(**run_kwargs)
        self.u_range = (-10, 5)
        self.w_range = (-10, 5)
        # hack, remove the x at the edges
        for d in ('U', 'W', 'T'):
            arr = getattr(self.r, d)
            setattr(self, d, arr[:, 15:-15, :])

        self.T_width = t_width
        self.Uf = self.reshape_to_current_relative(self.U, -50, self.T_width)
        self.Wf = self.reshape_to_current_relative(self.W, -50, self.T_width)
        # colour levels
        self.levels = np.linspace(*self.u_range, num=100)

    def reshape_to_current_relative(self, vel, T0, T1):
        """Take the velocity data and transform it to the current
        relative frame.
        """
        # tf is the time of front passage as f(x)
        tf = front_detect(self.U)
        # reshape, taking a constant T time intervals behind front
        X = np.indices((vel.shape[1],)).squeeze()
        U_ = np.dstack(vel[:, x, int(tf(x)) + T0:int(tf(x)) + T1] for x in X)
        # reshape this to same dimensions as before
        Uf = np.transpose(U_, (0, 2, 1))
        return Uf

    def plot_histogram_U(self, save=True):
        U = self.U
        uf = U.flatten()
        fig, ax = plt.subplots(nrows=2)

        ax[0].hist(uf, bins=1000, range=self.u_range)
        title = 'Streamwise velocity distribution, run {run}'
        ax[0].set_title(title.format(run=self.index))
        ax[0].set_xlabel('Streamwise velocity, pixels')
        ax[0].set_ylabel('frequency')
        ax[0].set_yticks([])

        Z = np.indices((U.shape[0],)).squeeze()
        U_bins = np.histogram(U[0], range=self.u_range, bins=1000)[1]
        histograms = (np.histogram(U[z], range=self.u_range, bins=1000)[0]
                      for z in xrange(U.shape[0]))
        hist_z = np.dstack(histograms).squeeze()

        levels = np.linspace(0, 500, 100)
        c1 = ax[1].contourf(U_bins[1:], Z, hist_z.T, levels=levels)
        # fig.colorbar(c1, ax=ax[1], use_gridspec=True)

        ax[1].set_title('velocity distribution')
        ax[1].set_xlabel('Streamwise velocity')
        ax[1].set_ylabel('z')

        fname = 'histogram_U_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)

        fig.tight_layout()

        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_histogram_W(self, save=True):
        U = self.W
        uf = U.flatten()
        fig, ax = plt.subplots(nrows=2)

        ax[0].hist(uf, bins=1000, range=self.u_range)
        title = 'Vertical velocity distribution, run {run}'
        ax[0].set_title(title.format(run=self.index))
        ax[0].set_xlabel('Vertical velocity, pixels')
        ax[0].set_ylabel('frequency')
        ax[0].set_yticks([])

        Z = np.indices((U.shape[0],)).squeeze()
        U_bins = np.histogram(U[0], range=self.u_range, bins=1000)[1]
        histograms = (np.histogram(U[z], range=self.u_range, bins=1000)[0]
                      for z in xrange(U.shape[0]))
        hist_z = np.dstack(histograms).squeeze()

        levels = np.linspace(0, 500, 100)
        c1 = ax[1].contourf(U_bins[1:], Z, hist_z.T, levels=levels)
        # fig.colorbar(c1, ax=ax[1], use_gridspec=True)

        ax[1].set_title('velocity distribution')
        ax[1].set_xlabel('vertical velocity')
        ax[1].set_ylabel('z')

        fname = 'histogram_W_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)

        fig.tight_layout()

        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_average_velocity(self, save=True):
        u_mod = np.hypot(self.U, self.W)
        u_mod_bar = stats.nanmean(u_mod, axis=2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        contourf = ax.contourf(u_mod_bar, self.levels)
        fig.colorbar(contourf)
        ax.set_title(r'Mean speed $\overline{|u|_t}(x, z)$')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        fname = 'average_velocity_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)
        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_median_velocity(self, save=True):
        u_mod = np.hypot(self.U, self.W)
        u_mod_med = stats.nanmedian(u_mod, axis=2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(u_mod_med, self.levels)
        ax.set_title('Median speed')
        ax.set_xlabel('horizontal')
        ax.set_ylabel('vertical')
        fname = 'median_velocity_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)
        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_power(self, save=True):
        # fourier transform over the time axis
        fig = plt.figure()
        ax_location = fig.add_subplot(211)
        ax_power = fig.add_subplot(212)

        space_mean = stats.nanmean(self.Uf, axis=1)
        ax_location.contourf(space_mean, self.levels)
        ax_location.set_title('Overview')

        times = self.T

        # compute fft over time
        power_spectrum = np.abs(np.fft.fft(self.U, axis=2)) ** 2
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

        ax_power.pcolormesh(X, Y, mhh.T, cmap='jet',
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
        ax_power.plot(x53, y53, 'k', linewidth=2)
        ax_power.text(x0, y0, '-5/3')

        ax_power.set_title('Power Spectrum')
        ax_power.set_xscale('log')
        ax_power.set_yscale('log')
        ax_power.set_xlim(xlo, xhi)
        ax_power.set_ylim(ylo, yhi)
        ax_power.set_xlabel('Frequency (Hz)')
        ax_power.set_ylabel('Power')

        fig.tight_layout()

        fname = 'power_spectrum_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)
        # TODO: can a decorator replace this functionality?
        if save:
            fig.savefig(fpath)
        elif not save:
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

    def plot_autocorrelation(self, save=True):
        """Plot the autocorrelation as a function of height of
        the mean front relative frame.
        """
        U = stats.nanmean(self.U, axis=1)
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

        fname = 'autocorrelation_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)
        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def time_slice_path(self, t):
        fname = 'time_slices/time_slice_{r}_{t:0>4d}.png'
        fpath = os.path.join(plot_dir, fname.format(r=self.index, t=t))
        return fpath

    def vertical_transect_path(self, x):
        fname = 'vertical_transects/vertical_transect_{r}_{x:0>4d}.png'
        fpath = os.path.join(plot_dir, fname.format(r=self.index, x=x))
        return fpath

    def main(self):
        # print "U distribution...",
        # self.plot_histogram_U()
        # print "W distribution...",
        # self.plot_histogram_W()
        # print "mean velocity...",
        # self.plot_average_velocity()
        # print "median velocity...",
        # self.plot_median_velocity()
        print "power spectrum...",
        self.plot_power()
        # print "autocorrelation..."
        # self.plot_autocorrelation()
        # print "vertical transects..."
        # self.plot_vertical_transects()
        # print "time slices..."
        # self.plot_time_slices()


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


if __name__ == '__main__':
    test_run_index = '3ban2y82'
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        help="single run test mode",
                        nargs='?',
                        const=test_run_index)
    parser.add_argument("--reload",
                        help="force reloading cache files",
                        action='store_true')
    args = parser.parse_args()

    if os.environ['HOSTNAME'] != 'doug-and-duck':
        print "Not running on doug-and-duck, are you sure?"
        raw_input()

    if args.test:
        r = test_run(index=test_run_index, reload=args.reload)
        # calling PlotRun loads everything anyway so can
        # call save here
        r.r.save()
        r.main()

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
            pr.main()
