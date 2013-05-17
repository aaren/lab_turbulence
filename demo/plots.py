import os

import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from gc_turbulence.gc_turbulence.turbulence import SingleLayer2dRun
from gc_turbulence.gc_turbulence.util import parallel_process
# from gc_turbulence import SingleLayer2dRun


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

# run for testing
test_run = '3ban2y82'
wd = os.path.join(w_dir, test_run)
run_kwargs = {'data_dir':  wd,
              'ffmt':      'img*csv',
              'parallel':  True,
              'limits':    run_lims[test_run]}
r = SingleLayer2dRun(**run_kwargs)


class PlotRun(object):
    def __init__(self, run):
        self.index = run
        wd = os.path.join(w_dir, run)
        run_kwargs = {'data_dir':  wd,
                      'ffmt':      'img*csv',
                      'parallel':  True,
                      'limits':    run_lims[run]}
        self.r = SingleLayer2dRun(**run_kwargs)

    def plot_histogram_U(self, save=True):
        U = self.r.U
        uf = U.flatten()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(uf, bins=1000, range=(-7, 3))
        title = 'Streamwise velocity distribution, run {run}'
        ax.set_title(title.format(run=run))
        ax.set_xlabel('Streamwise velocity, pixels')
        ax.set_ylabel('frequency')
        ax.set_yticks([])

        fname = 'histogram_U_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)

        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_histogram_W(self, save=True):
        U = self.r.W
        uf = U.flatten()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(uf, bins=1000, range=(-7, 3))
        title = 'Vertical velocity distribution, run {run}'
        ax.set_title(title.format(run=run))
        ax.set_xlabel('Vertical velocity, pixels')
        ax.set_ylabel('frequency')
        ax.set_yticks([])

        fname = 'histogram_W_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)

        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_average_velocity(self, save=True):
        modu = self.r.average_velocity()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(modu, 50)
        ax.set_title('Mean speed')
        fname = 'average_velocity_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)
        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_median_velocity(self, save=True):
        u_mod = np.hypot(self.r.U, self.r.W)
        u_mod_med = stats.nanmedian(u_mod, axis=2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(u_mod_med, 50)
        ax.set_title('Median speed')
        fname = 'median_velocity_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)
        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

    def plot_power(self, save=True):
        # fourier transform over the time axis
        U_point = np.abs(self.r.U[15, 30, :])

        fig = plt.figure()
        ax_location = fig.add_subplot(311)
        ax_speed = fig.add_subplot(312)
        ax_power = fig.add_subplot(313)

        ax_location.contourf(self.r.U[:, :, 50], 50)
        ax_location.set_title('Overview')
        ax_location.plot(30, 15, 'wo')

        # ???
        times = np.array(range(len(U_point))) * 0.01

        ax_speed.plot(times, U_point)
        ax_speed.set_title('Absolute streamwise velocity')
        ax_speed.set_xlabel('Time (s)')
        ax_speed.set_ylabel('Streamwise velocity (pixels)')

        power_spectrum = np.abs(np.fft.fft(U_point)) ** 2
        freqs = np.fft.fftfreq(len(U_point), d=0.01)

        ax_power.plot(freqs, power_spectrum, 'r.')
        ax_power.set_title('Power Spectrum')
        ax_power.set_yscale('log')
        ax_power.set_xlim(0, 50)
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

    def plot_vertical_transects(self):
        """If we take a vertical profile at a given horizontal
        position we can make a contour plot of some quantity over
        the vertical axis and time.

        Similarly, for a given moment in time we could make a contour
        plot over the horizontal and vertical axes.

        If we make a plots over all of the possible vertical profiles,
        we end up with a series of plots that can be animated.
        """
        X = range(self.r.U.shape[1])[::-1]
        arglist = [dict(x=x,
                        index=self.index,
                        U=self.r.U,
                        fname=self.fig_path(x))
                   for x in X]
        parallel_process(plot_vertical_transect, arglist)

    def fig_path(self, x):
        fname = 'vertical_transects/vertical_transect_{r}_{x:0>4d}.png'
        fpath = os.path.join(plot_dir, fname.format(r=self.index, x=x))
        return fpath


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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = 'Vertical transect {r} {x:0>4d}'.format(r=index, x=x)
    ax.set_title(title)
    U = U[:, x, :]
    ax.contourf(U, 100)
    fig.savefig(fname)
    queue.put(1)
    pbar.update()


if __name__ == '__main__':
    for run in runs:
        print "Plotting " + run + "..."
        pr = PlotRun(run)
        pr.plot_histogram_U()
        pr.plot_histogram_W()
        pr.plot_average_velocity()
        pr.plot_median_velocity()
        pr.plot_power()
        print "plotting vertical transects..."
        pr.plot_vertical_transects()
