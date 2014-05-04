import numpy as np
import matplotlib.pyplot as plt

import gc_turbulence as g
import sparse_dmd as dmd


def plot(index):
    cache_path = g.default_processed + index + '.hdf5'
    print cache_path,
    r = g.ProcessedRun(cache_path=cache_path)

    # make a plot with the average front relative streamwise velocity
    # compare the dim and non dim

    # TODO: add streamwise (non fr) velocity (vertical mean) through
    # time (line plot)
    # Trying to see how wavy runs are.
    # add this in a wide plot below the other two

    u_levels = np.linspace(-0.12, 0.035, 200)
    u_levels_ = u_levels * 7

    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(231)
    ax1 = fig.add_subplot(232)
    ax1a = fig.add_subplot(233)
    ax3 = fig.add_subplot(212)

    fig.suptitle('{}, front_speed={}'.format(index, r.front_speed[...]), y=1.0)

    mean = np.mean(r.Uf[:], axis=1)
    mean_ = np.mean(r.Uf_[:], axis=1)

    mean_sub = r.Uf[:, 10, :] - mean

    z = r.Zf[:, 0, :]
    t = r.Tf[:, 0, :]

    z_ = r.Zf_[:, 0, :]
    t_ = r.Tf_[:, 0, :]

    ax0.contourf(t, z, mean, levels=u_levels)
    ax1.contourf(t_, z_, mean_, levels=u_levels_)
    ax1a.contourf(t, z, mean_sub, levels=u_levels)

    ax0.set_xlim(-5, 20)
    ax0.set_ylim(0, 0.125)
    ax0.set_title('front relative mean')

    ax1a.set_xlim(-5, 20)
    ax1a.set_ylim(0, 0.125)
    ax1a.set_title('mean subtracted ix=10')

    ax1.set_xlim(-3, 12)
    ax1.set_ylim(0, 0.5)
    ax1.set_title('non dim front relative mean')

    vertical_mean = np.mean(r.U[:, 10, :], axis=0)
    time = r.T[0, 10, :]

    ax3.plot(time, vertical_mean, label='vertical mean')
    ax3.set_title('vertical mean')

    fig.tight_layout()
    fig.savefig('compare/{}.png'.format(index))

    plt.close(fig)


def plot_all():
    # todo: do this for all single layer runs
    indices = ['r13_12_09a',
               'r13_12_09b',
               'r13_12_09c',
               'r13_12_09d',
               'r13_12_09e',
               'r13_12_09f',
               'r13_12_16a',
               'r13_12_16b',
               'r13_12_16c',
               'r13_12_16d',
               'r13_12_16e',
               'r13_12_16f',
               'r13_12_17a',
               'r13_12_17c',
               'r13_12_17d',
               'r13_12_17e',
               'r14_01_14a',
               'r14_01_14b',
               'r14_01_14c',
               'r14_01_14d',
               'r14_01_14e',
               'r14_01_14f',
               'r14_01_16f']

    for index in indices:
        print "\rplotting ", index,
        try:
            plot(index)
        except:
            print "failed", index


def dmd_stack():
    index = 'r14_01_14a'
    r = g.ProcessedRun(cache_path=g.default_processed + index + '.hdf5')

    u = r.Uf[:]
    v = r.Vf[:]
    w = r.Wf[:]

    uvw = np.concatenate([x[..., None] for x in u, v, w], axis=-1)

    su = dmd.to_snaps(u, decomp_axis=1)
    sa = dmd.to_snaps(uvw, decomp_axis=1)

    udmd = dmd.SparseDMD(su)
    admd = dmd.SparseDMD(sa)

    gammaval = np.logspace(0, 5, 100)

    udmd.compute_dmdsp(gammaval)
    admd.compute_dmdsp(gammaval)


def dmd_spiral():
    index = 'r14_01_14a'
    r = g.ProcessedRun(cache_path=g.default_processed + index + '.hdf5')

    u = r.Uf[:]
    su = dmd.to_snaps(u, decomp_axis=1)

    udmd = dmd.SparseDMD(su)

    gammaval = np.logspace(0, 5, 100)

    udmd.compute_dmdsp(gammaval)

    udmd.compute_sparse_reconstruction(Ni=30, data=u, decomp_axis=1)

    a1 = udmd.reconstruction.amplitudes[1]
    a2 = udmd.reconstruction.amplitudes[3]

    u1 = udmd.reconstruction.freqs[1]
    u2 = udmd.reconstruction.freqs[3]

    # need this for 3d projections
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('data trajectory projected into first '
                 'two non mean dynamic modes')

    # plot time as vertical axis and projection onto
    # first two modes as horizontal axes
    t = np.linspace(0, 300, 100)
    m1 = a1 * u1 ** t
    m2 = a2 * u2 ** t

    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_zlabel('t')
    ax.plot(m1, m2, t)


if __name__ == '__main__':
    plot_all()
