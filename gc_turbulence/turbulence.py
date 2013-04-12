# read in a piv text dump

import glob
from multiprocessing.pool import Pool

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

# single frame


def determine_shape(fname):
    """Get the gridsize from a piv text file by filtering
    the metadata in the header.
    """
    with open(fname) as f:
        content = [l.strip('\r\n') for l in f.readlines()]
        gridsize = content[3].split(':')[1]
        x = gridsize.split(', ')[0][-2:]
        y = gridsize.split(', ')[1][-3:-1]
        shape = (int(y), int(x))
    return shape


def get_data(fname='data/Export.3atnh4dp.000500.txt'):
    # extract data
    D = np.genfromtxt(fname, skip_header=9)
    shape = determine_shape(fname)

    # reshape to sensible
    X = D[:, 0].reshape(shape)
    Y = D[:, 1].reshape(shape)
    U = D[:, 6].reshape(shape)
    V = D[:, 7].reshape(shape)

    return X, Y, U, V


def gen_quiver_plot(fname):
    fig = plt.figure()
    ax = plt.axes(xlim=(10, 80), ylim=(0, 50))
    x, y, u, v = get_data(fname)
    ax.quiver(u, v, scale=200)
    new_name = fname.split('.')[-2]
    fig.savefig('quiver/quiver_{f}.png'.format(f=new_name))


def animate_quiver():
    """Follows the animation tutorial at
    http://jakevdp.github.com/blog/2012/08/18/matplotlib-animation-tutorial/
    """
    files = glob.glob('data/Export*txt')

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 100), ylim=(0, 70))
    quiver = ax.quiver([], [])

    def init():
        quiver.set_UVC([], [])
        return quiver

    def animate(i):
        fname = files[i]
        x, y, u, v = get_data(fname)
        quiver.set_UVC(u, v)
        return quiver

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(files), interval=20,
                                   blit=True)

    print("Creating the animation...")
    anim.save('quiver_animation.mp4', fps=30)


if __name__ == '__main__':
    # animate_quiver()
    files = glob.glob('data/Export*txt')
    # files = glob.glob('data/Export.3atnh4dp.0005*txt')
    p = Pool()
    p.map(gen_quiver_plot, files)
    p.close()
    p.join()
