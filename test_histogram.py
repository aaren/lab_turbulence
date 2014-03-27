import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import gc_turbulence as g


plt.rc('figure', figsize=(15, 15))

t_bins = np.linspace(-5, 20, 500)
levels = np.linspace(0, 20)

test_cache = g.default_processed + '../test/r13_12_17c.hdf5'

r = g.ProcessedRun(cache_path=test_cache)

u_bins = np.linspace(-0.1, 0.05, 100)
w_bins = np.linspace(-0.05, 0.05, 100)
v_bins = w_bins


def plot_time_histogram():
    iz = 30
    multi_point = np.s_[iz, :, :]

    u_data = {'name': ('time after front passage',
                       'streamwise velocity'),
              'data': (r.Tf[multi_point].flatten(),
                       r.Uf[multi_point].flatten())}
    v_data = {'name': ('time after front passage',
                       'cross-stream velocity'),
              'data': (r.Tf[multi_point].flatten(),
                       r.Vf[multi_point].flatten())}
    w_data = {'name': ('time after front passage',
                       'vertical velocity'),
              'data': (r.Tf[multi_point].flatten(),
                       r.Wf[multi_point].flatten())}

    fig, axes = plt.subplots(nrows=3)

    fig.suptitle('velocity pdf as a function of time')

    g.Histograms.plot_histogram(axes[0], data=u_data,
                                bins=(t_bins, u_bins),
                                levels=levels)
    g.Histograms.plot_histogram(axes[1], data=v_data,
                                bins=(t_bins, v_bins),
                                levels=levels)
    g.Histograms.plot_histogram(axes[2], data=w_data,
                                bins=(t_bins, w_bins),
                                levels=levels)

    fig.tight_layout()

    fig.savefig('histogram_test.png')


def plot_covariance():
    uv_data = {'name': ('streamwise velocity',
                        'cross-stream velocity'),
               'data': (r.Uf[:].flatten(),
                        r.Vf[:].flatten())}
    uw_data = {'name': ('streamwise velocity',
                        'vertical velocity'),
               'data': (r.Uf[:].flatten(),
                        r.Wf[:].flatten())}
    vw_data = {'name': ('cross-stream velocity',
                        'vertical velocity'),
               'data': (r.Vf[:].flatten(),
                        r.Wf[:].flatten())}

    fig, axes = plt.subplots(nrows=3)

    fig.suptitle('velocity covariance')

    levels = np.logspace(0, 4, 100)

    g.Histograms.plot_histogram(axes[0], data=uv_data,
                                bins=(u_bins, v_bins),
                                levels=levels,
                                norm=mpl.colors.LogNorm())
    g.Histograms.plot_histogram(axes[1], data=uw_data,
                                bins=(u_bins, w_bins),
                                levels=levels,
                                norm=mpl.colors.LogNorm())
    g.Histograms.plot_histogram(axes[2], data=vw_data,
                                bins=(v_bins, w_bins),
                                levels=levels,
                                norm=mpl.colors.LogNorm())

    fig.savefig('covariance.png')


def test():
    plot_time_histogram()
    plot_covariance()


if __name__ == '__main__':
    test()
