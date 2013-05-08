import os

from gc_turbulence.gc_turbulence.turbulence import SingleLayer2dRun

import matplotlib.pyplot as plt


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


class PlotRun(object):
    def __init__(self, run):
        self.index = run
        # wd = os.path.join(w_dir, run)
        wd = w_dir + run + '/'
        run_kwargs = {'data_dir':  wd,
                      'ffmt':      'img*csv',
                      'parallel':  True,
                      'limits':    run_lims[run]}
        self.r = SingleLayer2dRun(**run_kwargs)

    def plot_histogram(self, save=True):
        U = self.r.U
        uf = U.flatten()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(uf, bins=1000, range=(-7, 3))
        fname = 'histogram_' + self.index + '.png'
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
        fname = 'average_velocity_' + self.index + '.png'
        fpath = os.path.join(plot_dir, fname)
        if save:
            fig.savefig(fpath)
        elif not save:
            return fig

if __name__ == '__main__':
    for run in runs:
        print "Plotting " + run + "..."
        pr = PlotRun(run)
        pr.plot_histogram()
        pr.plot_average_velocity()
