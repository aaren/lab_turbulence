import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import gc_turbulence as g


plt.rc('figure', figsize=(15, 15))

t_bins = np.linspace(-5, 20, 500)
levels = np.linspace(0, 20)

test_cache = g.default_processed + 'r13_12_16a.hdf5'

r = g.ProcessedRun(cache_path=test_cache)

u_bins = np.linspace(-0.15, 0.05, 100)
w_bins = np.linspace(-0.05, 0.05, 100)
v_bins = w_bins
abs_bins = np.linspace(0, 0.15, 100)

z_bins = np.linspace(0.05, 0.12, 30)


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


def plot_vertical_time_histogram(ydata, y_bins, ylabel, outdir,
                                 sub_region=np.s_[:]):
    bins = (t_bins, y_bins, z_bins)
    data = r.Tf[sub_region].flatten(), ydata, r.Zf[sub_region].flatten()

    H, edges = np.histogramdd(data, bins=bins, normed=True)
    xedges, yedges = edges[:2]
    Hmasked = np.ma.masked_where(H == 0, H)

    # levels = np.logspace(0, 2.5)
    indices = np.arange(z_bins.size - 1)

    for iz in indices:
        print iz, "\r",

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.contourf(xedges[1:], yedges[1:], Hmasked.T[iz], 100)
                    # levels=levels, norm=mpl.colors.LogNorm())
        ax.set_title("z = {}".format(z_bins[iz]))
        ax.set_xlabel('time after front passage')
        ax.set_ylabel(ylabel)

        fig.savefig('{}/{:03d}.png'.format(outdir, iz))
        plt.close(fig)


def plot_vertical_covariance_histogram(xdata, ydata,
                                       x_bins, y_bins,
                                       xlabel, ylabel,
                                       outdir, sub_region=np.s_[:]):
    """Plot the covariance of x and y over multiple heights."""
    bins = (x_bins, y_bins, z_bins)
    data = xdata, ydata, r.Zf[sub_region].flatten()

    H, edges = np.histogramdd(data, bins=bins, normed=True)
    xedges, yedges = edges[:2]
    Hmasked = np.ma.masked_where(H == 0, H)

    levels = np.logspace(0, 4.5)
    indices = np.arange(z_bins.size - 1)

    for iz in indices:
        print iz, "\r",

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.contourf(xedges[1:], yedges[1:], Hmasked.T[iz],
                    norm=mpl.colors.LogNorm(), levels=levels)
        ax.set_title("z = {}".format(z_bins[iz]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        g.util.makedirs_p(outdir)
        fout = '{}/{:03d}.png'.format(outdir, iz)
        fig.savefig(fout)
        plt.close(fig)


def plot_multi_time_histograms():
    sub = np.s_[:]
    abs_speed = np.sqrt(r.Uf[sub] ** 2 + r.Wf[sub] ** 2 + r.Vf[sub] ** 2)

    kwarglist = [dict(outdir='many_streamwise',
                      ylabel='streamwise velocity',
                      y_bins=u_bins,
                      ydata=r.Uf[sub].flatten(),
                      sub_region=sub),

                 dict(outdir='many_vertical',
                      ylabel='vertical velocity',
                      y_bins=w_bins,
                      ydata=r.Wf[sub].flatten(),
                      sub_region=sub),

                 dict(outdir='many_cross',
                      ylabel='cross stream velocity',
                      y_bins=v_bins,
                      ydata=r.Vf[sub].flatten(),
                      sub_region=sub),

                 dict(outdir='many_abs',
                      ylabel='absolute speed',
                      bins=abs_bins,
                      ydata=abs_speed.flatten(),
                      sub_region=sub),
                 ]
    for kwargs in kwarglist:
        plot_vertical_time_histogram(**kwargs)


def plot_multi_covariance_histograms():
    sub = np.s_[:]

    plot_kwargs = [dict(outdir='many_covar_uv',
                        xlabel='streamwise velocity',
                        ylabel='cross stream velocity',
                        x_bins=u_bins,
                        y_bins=v_bins,
                        xdata=r.Uf[sub].flatten(),
                        ydata=r.Vf[sub].flatten(),
                        sub_region=sub),

                   dict(outdir='many_covar_uw',
                        xlabel='streamwise velocity',
                        ylabel='vertical velocity',
                        x_bins=u_bins,
                        y_bins=w_bins,
                        xdata=r.Uf[sub].flatten(),
                        ydata=r.Wf[sub].flatten(),
                        sub_region=sub),

                   dict(outdir='many_covar_vw',
                        xlabel='cross stream velocity',
                        ylabel='vertical velocity',
                        x_bins=v_bins,
                        y_bins=w_bins,
                        xdata=r.Vf[sub].flatten(),
                        ydata=r.Wf[sub].flatten(),
                        sub_region=sub),
                   ]

    for kwargs in plot_kwargs:
        plot_vertical_covariance_histogram(**kwargs)


def test():
    # plot_time_histogram()
    # plot_covariance()
    # plot_multi_time_histograms()
    plot_multi_covariance_histograms()


if __name__ == '__main__':
    test()
