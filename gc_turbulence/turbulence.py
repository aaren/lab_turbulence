import glob
from multiprocessing.pool import Pool

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import stats


def lazyprop(fn):
    """Decorator to allow lazy evaluation of class properties

    http://stackoverflow.com/questions/3012421/python-lazy-property-decorator

    usage:

        class Test(object):

            @lazyprop
            def a(self):
                print 'generating "a"'
                return range(5)

    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


class SingleLayer2dFrame(object):
    """Each SingleLayer2dRun is comprised of a series of frames.
    This class represents one of the frames.
    """
    def __init__(self, fname, quiver_format='quiver/quiver_{f}.png'):
        """Initialise a frame object.

        Inputs: fname - filename of a piv velocity text file
                quiver_format - the format of quiver file output
        """
        self.fname = fname
        self.quiver_format = quiver_format

    @lazyprop
    def header(self):
        """Pull header from velocity file and return as dictionary."""
        with open(self.fname) as f:
            content = f.read().splitlines()
            head = content[1:7]
            header_info = {}
            for h in head:
                k = h.split(':')[0]
                v = ':'.join(h.split(':')[1:])
                header_info[k] = v
        return header_info

    @lazyprop
    def shape(self):
        """Get the gridsize from a piv text file by filtering
        the metadata in the header.
        """
        gridsize = self.header['GridSize']
        x = gridsize.split(', ')[0][-2:]
        y = gridsize.split(', ')[1][-3:-1]
        shape = (int(y), int(x))
        return shape

    @lazyprop
    def data(self, delimiter=None):
        """Extract data from a PIV velocity text file."""
        if not delimiter:
            # force determine delimiter
            if self.header['FileID'] == 'DSExport.CSV':
                delimiter = ','
            elif self.header['FileID'] == 'DSExport.TAB':
                delimiter = None
        # extract data
        # TODO: dtypes
        D = np.genfromtxt(self.fname, skip_header=9, delimiter=delimiter)
        shape = self.shape

        # reshape to sensible
        X = D[:, 0].reshape(shape)
        Y = D[:, 1].reshape(shape)
        U = D[:, 6].reshape(shape)
        V = D[:, 7].reshape(shape)

        return dict(X=X, Y=Y, U=U, V=V)

    # TODO: must be a way to do this programmatically, setattr?
    @lazyprop
    def x(self):
        return self.data['X']

    @lazyprop
    def y(self):
        return self.data['Y']

    @lazyprop
    def u(self):
        return self.data['U']

    @lazyprop
    def v(self):
        return self.data['V']

    def gen_quiver_plot(self, fig=None):
        """Make a quiver plot of the frame data."""
        if not fig:
            fig = plt.figure()
        ax = plt.axes(xlim=(10, 80), ylim=(0, 50))
        ax.quiver(self.u, self.v, scale=200)
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
    """Represents a 2D PIV experiment. Written for a gravity current
    generated by lock release impinging on a homogeneous ambient, but
    perhaps can be made more general.
    """
    def __init__(self, data_dir='data/', ffmt='Export*txt'):
        """Initialise a run.

        Inputs: data_dir - directory containing the velocity files
                ffmt - format of the velocity filenames (passed to glob)
        """
        self.files = glob.glob(data_dir + ffmt)
        self.quiver_format = 'quiver/quiver_{f}.png'

    @lazyprop
    def frames(self):
        """List of frame objects corresponding to the input files. """
        frames = [SingleLayer2dFrame(f) for f in self.files]
        return frames

    # TODO: programmatically, setattr
    @lazyprop
    def V(self):
        return np.dstack([f.v for f in self.frames])

    @lazyprop
    def U(self):
        return np.dstack([f.u for f in self.frames])

    @staticmethod
    def gen_quiver_plot(fname):
        """Create a frame instance and make the quiver plot.

        This is a wrapper to allow multiprocessing.
        """
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
            x, y, u, v = frame.data
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
        in parallel using multiprocessing.
        """
        p = Pool()
        p.map(self.gen_quiver_plot, self.files)
        p.close()
        p.join()

    def average_velocity(self):
        """Return the time averaged velocity over the run domain."""
        u_mod = np.hypot(self.U, self.V)
        u_mod_bar = stats.nanmean(u_mod, axis=2)
        # plt.contourf(u_bar, 100)
        return u_mod_bar

    def std_velocity(self):
        """Return the standard deviation of the absolute velocity
        over the run domain."""
        u_mod = np.hypot(self.U, self.V)
        u_mod_std = stats.nanstd(u_mod, axis=2)
        return u_mod_std


if __name__ == '__main__':
    # TODO: move to tests
    r = SingleLayer2dRun()
    r.make_quivers()
