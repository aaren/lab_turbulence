import os
import glob

import numpy as np
from scipy import stats

if 'DISPLAY' not in os.environ:
    import matplotlib as mpl
    mpl.use('Agg')

import matplotlib.pyplot as plt

from util import parallel_process
from util import makedirs_p
from util import ProgressBar


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
    def __init__(self, fname, quiver_format='quiver_{f}.png'):
        """Initialise a frame object.

        Inputs: fname - filename of a piv velocity text file
                quiver_format - the format of quiver file output
        """
        self.fname = fname
        self.quiver_format = quiver_format
        # TODO: must be a way to do this programmatically, setattr?
        self.x = self.data['X']
        self.z = self.data['Z']
        self.u = self.data['U']
        self.w = self.data['W']

    @property
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

    @property
    def shape(self):
        """Get the gridsize from a piv text file by filtering
        the metadata in the header.
        """
        gridsize = self.header['GridSize']
        x = gridsize.split(', ')[0][-2:]
        z = gridsize.split(', ')[1][-3:-1]
        shape = (int(z), int(x))
        return shape

    @property
    def data(self, delimiter=None):
        """Extract data from a PIV velocity text file.

        N.B. Here I've used the convention (u, w) for (streamwise,
        vertical) velocity, in contrast to the files which use (u, v).
        Similarly for (x, z) rather than (x, y).

        This is to be consistent with meteorological convention.
        """
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
        Z = D[:, 1].reshape(shape)
        U = D[:, 6].reshape(shape)
        W = D[:, 7].reshape(shape)

        return dict(X=X, Z=Z, U=U, W=W)

    def make_quiver_plot(self, quiver_dir=''):
        """Make a quiver plot of the frame data."""
        fig = plt.figure()
        ax = plt.axes(xlim=(10, 80), ylim=(0, 50))
        ax.quiver(self.u, self.w, scale=200)
        quiver_name = self.quiver_name
        quiver_path = quiver_dir + quiver_name
        fig.savefig(quiver_path)

    @property
    def quiver_name(self):
        """Generate name for quiver plot given an input text
        filename.
        """
        new_name = self.fname.split('.')[-2]
        quiver_fname = self.quiver_format.format(f=new_name)
        return quiver_fname


# These functions are out here because they need to be pickleable
# for multiprocessing to work. Shame of this is that they can't
# access class state.
def instantiateFrame(args):
    """Create and return a frame instance. Single argument which
    consists of a filename and a queue. The filename is the file
    to create the frame from (as in __init__ of SingleLayer2dFrame).

    the queue is used to store the output

    the pbar is a progress bar that is shared between multiple
    Process() instances.
    """
    fname = args['fname']
    queue = args['queue']
    pbar = args['pbar']
    frame = SingleLayer2dFrame(fname)
    pbar.update()
    queue.put(frame)
    return


def gen_quiver_plot(args):
    """Create a frame instance and make the quiver plot.

    This is a wrapper to allow multiprocessing.
    """
    fname = args['fname']
    queue = args['queue']
    pbar = args['pbar']
    quiver_dir = args['quiver_dir']
    frame = SingleLayer2dFrame(fname)
    frame.make_quiver_plot(quiver_dir=quiver_dir)
    pbar.update()
    queue.put(1)


class SingleLayer2dRun(object):
    """Represents a 2D PIV experiment. Written for a gravity current
    generated by lock release impinging on a homogeneous ambient, but
    perhaps can be made more general.
    """
    def __init__(self, data_dir='data/', ffmt='Export*txt',
                 parallel=True, limits=None):
        """Initialise a run.

        Inputs: data_dir - directory containing the velocity files
                ffmt - format of the velocity filenames (passed to glob)
                parallel - whether to use multiprocessing (default True)
                limits - (start, finish) frame indices to use in selecting
                         the list of files. Default None is to use all
                         the files.
        """
        self.data_dir = data_dir
        self.allfiles = sorted(glob.glob(data_dir + 'data/' + ffmt))
        if limits:
            first, last = limits
            self.files = self.allfiles[first:last]
        else:
            self.files = self.allfiles
        self.nfiles = len(self.files)
        self.quiver_dir = data_dir + 'quiver/'
        self.quiver_format = 'quiver_{f}.png'
        self.parallel = parallel

    @lazyprop
    def frames(self):
        """List of frame objects corresponding to the input files. """
        if self.parallel:
            frames = self.get_frames_parallel()
        elif not self.parallel:
            frames = self.get_frames_serial()
        return frames

    def get_frames_serial(self):
        """Get the frames without multiprocessing. We could just do
        something simple like

            frames = [SingleLayer2dFrame(f) for f in self.files]

        but if we want a progress bar we need to be more explicit.

        Doesn't follow the same pattern as `get_frames_parallel`
        but probably could if we used a queue.
        """
        frames = []
        nfiles = len(self.files)
        pbar = ProgressBar(maxval=nfiles)
        pbar.start()
        for i, fname in enumerate(self.files):
            frame = SingleLayer2dFrame(fname)
            frames.append(frame)
            pbar.update(i)
        pbar.finish()
        return frames

    def get_frames_parallel(self, processors=20):
        """Get the frames with multiprocessing.
        """
        args = [dict(fname=f) for f in self.files]
        frames = parallel_process(instantiateFrame, args, processors)
        if type(frames) is not list:
            raise UserWarning('frames is not list!')
        # order based on filename
        sorted_frames = sorted(frames, key=lambda f: f.fname)
        return sorted_frames

    # TODO: programmatically, setattr
    @lazyprop
    def W(self):
        return np.dstack(f.w for f in self.frames)

    @lazyprop
    def U(self):
        return np.dstack(f.u for f in self.frames)

    def make_quivers(self):
        """Take the text files for this run and generate quiver plots
        in parallel using multiprocessing.
        """
        print("Generating {n} quiver plots".format(n=len(self.files)))
        makedirs_p(self.quiver_dir)
        arglist = [dict(fname=f, quiver_dir=self.quiver_dir) for f in self.files]
        parallel_process(gen_quiver_plot, arglist)

    def average_velocity(self):
        """Return the time averaged velocity over the run domain."""
        u_mod = np.hypot(self.U, self.W)
        u_mod_bar = stats.nanmean(u_mod, axis=2)
        # plt.contourf(u_bar, 100)
        return u_mod_bar

    def std_velocity(self):
        """Return the standard deviation of the absolute velocity
        over the run domain."""
        u_mod = np.hypot(self.U, self.W)
        u_mod_std = stats.nanstd(u_mod, axis=2)
        return u_mod_std
