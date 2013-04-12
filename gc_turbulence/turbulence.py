# read in a piv text dump

import glob
from multiprocessing.pool import Pool

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation


class SingleLayer2dFrame(object):
    def __init__(self, fname):
        self.fname = fname

    def determine_shape(self):
        """Get the gridsize from a piv text file by filtering
        the metadata in the header.
        """
        with open(self.fname) as f:
            content = [l.strip('\r\n') for l in f.readlines()]
            gridsize = content[3].split(':')[1]
            x = gridsize.split(', ')[0][-2:]
            y = gridsize.split(', ')[1][-3:-1]
            shape = (int(y), int(x))
        return shape

    def get_data(self):
        """Extract data from a PIV velocity text file."""
        # extract data
        D = np.genfromtxt(self.fname, skip_header=9)
        shape = self.determine_shape()

        # reshape to sensible
        X = D[:, 0].reshape(shape)
        Y = D[:, 1].reshape(shape)
        U = D[:, 6].reshape(shape)
        V = D[:, 7].reshape(shape)

        return X, Y, U, V

    def gen_quiver_plot(self, fig=None):
        """Make a quiver plot of the run data."""
        if not fig:
            fig = plt.figure()
        ax = plt.axes(xlim=(10, 80), ylim=(0, 50))
        x, y, u, v = self.get_data()
        ax.quiver(u, v, scale=200)
        quiver_name = self.quiver_name()
        fig.savefig(quiver_name)

    def quiver_name(self):
        """Generate name for quiver plot given an input text
        filename.
        """
        new_name = self.fname.split('.')[-2]
        quiver_fname = self.quiver_format.format(f=new_name)
        return quiver_fname


class SingleLayer2dRun(object):
    def __init__(self, wdir='data/', ffmt='Export*txt'):
        self.files = glob.glob(wdir + ffmt)
        self.quiver_format = 'quiver/quiver_{f}.png'

    @staticmethod
    def gen_quiver_plot(fname):
        frame = SingleLayer2dFrame(fname)
        frame.gen_quiver_plot()

    def animate_quiver(self):
        """Follows the animation tutorial at
        http://jakevdp.github.com/blog/2012/08/18/matplotlib-animation-tutorial/
        """
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 100), ylim=(0, 70))
        quiver = ax.quiver([], [])

        def init():
            quiver.set_UVC([], [])
            return quiver

        def animate(i):
            fname = self.files[i]
            frame = SingleLayer2dFrame(fname)
            x, y, u, v = frame.get_data()
            quiver.set_UVC(u, v)
            return quiver

        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init,
                                       frames=len(self.files),
                                       interval=20,
                                       blit=True)
        print("Creating the animation...")
        anim.save('quiver_animation.mp4', fps=30)

    def make_quivers(self):
        """Take the text files for this run and generate quiver plots
        in parallel.
        """
        p = Pool()
        p.map(self.gen_quiver_plot, self.files)
        p.close()
        p.join()

if __name__ == '__main__':
    r = SingleLayer2dRun()
    r.make_quivers()
