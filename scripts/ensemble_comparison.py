import os

import numpy as np
import matplotlib.pyplot as plt

from gc_turbulence import AnalysisRun


working_dir = '/home/eeaol/lab/data/flume2/main_data'
cache_dir = '/home/eeaol/lab/data/flume2/main_data/cache'

# TODO: flesh this file out
info_file = 'runs'

runs, hashes = np.genfromtxt(info_file, dtype=str, skip_header=1, unpack=True)

runs = ['r13_12_12c']
hashes = ['3n1n8hb6']


class Plotter(object):
    def __init__(self, run_index, run_hash):
        self.index = run_index
        self.hash = run_hash

        self.r = AnalysisRun(pattern=run_hash, cache_path=cache_dir)
        self.r.init_reshape()

        self.z = self.r.z[4:104, 15:130, :]
        self.wf = self.r.wf[4:104, 15:130, :]

    def contour_plot(self):
        fig, ax = plt.subplots()
        ax.contourf(self.wf.mean(axis=1), levels=np.linspace(-0.02, 0.03))
        fname = os.path.join('plots', 'contour_' + self.index + '.png')
        fig.savefig(fname)

    def vertical_pdf(self):
        fig, ax = plt.subplots()
        w_range = (-0.02, 0.04)
        nbins = 200
        bin_values = np.linspace(w_range[0], w_range[1], nbins)
        hist_z_100 = np.array([np.histogram(z, range=w_range,
                                            bins=nbins, density=True)[0]
                               for z in self.wf[:, :, 100]])
        heights = self.z[:, 0, 0]
        ax.contourf(bin_values, heights, hist_z_100, levels=np.linspace(0, 400))
        fname = os.path.join('plots', 'vertical_pdf_' + self.index + '.png')
        fig.savefig(fname)


if __name__ == '__main__':
    for index, hash in zip(runs, hashes):
        print index
        plotter = Plotter(index, hash)

        plotter.contour_plot()
        print "plotted countour"
        plotter.vertical_pdf()
        print "plotted pdf"
