import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import gc_turbulence as g


plt.rc('figure', figsize=(15, 15))

t_bins = np.linspace(-5, 20, 500)
levels = np.linspace(0, 20)

test_cache = g.default_processed + '../test/r13_12_17c.hdf5'

r = g.ProcessedRun(cache_path=test_cache)

u_bins = np.linspace(-0.15, 0.05, 100)
w_bins = np.linspace(-0.05, 0.05, 100)
v_bins = w_bins
abs_bins = np.linspace(0, 0.15, 100)

sub = np.s_[10:20, 40:50, :]
sub = np.s_[:]

z_bins = np.linspace(0.05, 0.12, 30)


# TODO: refactor to single function and use loop
def plot_many_time_streamwise_histogram():
    bins = (t_bins, u_bins, z_bins)
    data = r.Tf[sub].flatten(), r.Uf[sub].flatten(), r.Zf[sub].flatten()

    H, edges = np.histogramdd(data, bins=bins, normed=True)
    xedges, yedges = edges[:2]
    Hmasked = np.ma.masked_where(H == 0, H)

    # levels = np.logspace(0, 2.5)
    indices = np.arange(z_bins.size - 1)

    for iz in indices:
        print iz

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.contourf(xedges[1:], yedges[1:], Hmasked.T[iz], 100)
                    # levels=levels, norm=mpl.colors.LogNorm())
        ax.set_title("z = {}".format(z_bins[iz]))
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('front relative streamwise velocity')

        fig.savefig('many_streamwise/{:03d}.png'.format(iz))
        plt.close(fig)


def plot_many_time_vertical_histogram():
    bins = (t_bins, w_bins, z_bins)
    data = r.Tf[sub].flatten(), r.Wf[sub].flatten(), r.Zf[sub].flatten()

    H, edges = np.histogramdd(data, bins=bins, normed=True)
    xedges, yedges = edges[:2]
    Hmasked = np.ma.masked_where(H == 0, H)

    # levels = np.logspace(0, 2.5)
    indices = np.arange(z_bins.size - 1)

    for iz in indices:
        print iz

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.contourf(xedges[1:], yedges[1:], Hmasked.T[iz], 100)
                    # levels=levels, norm=mpl.colors.LogNorm())
        ax.set_title("z = {}".format(z_bins[iz]))
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('vertical velocity')

        fig.savefig('many_vertical/{:03d}.png'.format(iz))


def plot_many_time_cross_histogram():
    bins = (t_bins, v_bins, z_bins)
    data = r.Tf[sub].flatten(), r.Vf[sub].flatten(), r.Zf[sub].flatten()

    H, edges = np.histogramdd(data, bins=bins, normed=True)
    xedges, yedges = edges[:2]
    Hmasked = np.ma.masked_where(H == 0, H)

    # levels = np.logspace(0, 2.5)
    indices = np.arange(z_bins.size - 1)

    for iz in indices:
        print iz

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.contourf(xedges[1:], yedges[1:], Hmasked.T[iz], 100)
                    # levels=levels, norm=mpl.colors.LogNorm())
        ax.set_title("z = {}".format(z_bins[iz]))
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('cross stream velocity')

        fig.savefig('many_cross/{:03d}.png'.format(iz))


def plot_many_time_abs_histogram():
    bins = (t_bins, abs_bins, z_bins)

    speed = np.sqrt(r.Uf[sub] ** 2 + r.Wf[sub] ** 2 + r.Vf[sub] ** 2)

    data = r.Tf[sub].flatten(), speed.flatten(), r.Zf[sub].flatten()

    H, edges = np.histogramdd(data, bins=bins, normed=True)
    xedges, yedges = edges[:2]
    Hmasked = np.ma.masked_where(H == 0, H)

    # levels = np.logspace(0, 2.5)
    indices = np.arange(z_bins.size - 1)

    for iz in indices:
        print iz

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.contourf(xedges[1:], yedges[1:], Hmasked.T[iz], 100)
                    # levels=levels, norm=mpl.colors.LogNorm())
        ax.set_title("z = {}".format(z_bins[iz]))
        ax.set_xlabel('time after front passage')
        ax.set_ylabel('absolute speed')

        fig.savefig('many_abs/{:03d}.png'.format(iz))


def plot_time_histogram(iz=30):
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


def plot_3d():
    data = (r.Tf[:].flatten(), r.Zf[:].flatten(), r.Wf[:].flatten())
    z_bins = np.linspace(0, 0.1, 50)
    bins = (t_bins, z_bins, w_bins)
    H, edges = np.histogramdd(data, bins=bins, normed=True)

    from mayavi import mlab

    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(H),
                                     plane_orientation='x_axes',
                                     slice_index=10,)
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(H),
                                     plane_orientation='y_axes',
                                     slice_index=10,)

    mlab.outline()

    # crashes! segfault. running ipython --gui=wx --pylab=wx
    # also crashes when run as python script


def test():
    # plot_time_histogram()
    # plot_covariance()
    # plot_many_time_streamwise_histogram()
    # plot_many_time_vertical_histogram()
    # plot_many_time_cross_histogram()
    plot_many_time_abs_histogram()


if __name__ == '__main__':
    test()
