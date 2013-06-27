import os
import glob
import cPickle as pickle

import numpy as np

if 'DISPLAY' not in os.environ:
    import matplotlib as mpl
    mpl.use('Agg')

import matplotlib.pyplot as plt

from util import parallel_process
from util import makedirs_p
from util import ProgressBar


class SingleLayerFrame(object):
    """Each SingleLayerRun is comprised of a series of frames.
    This class represents one of the frames.
    """
    def __init__(self, fname, stereo=False, quiver_format='quiver_{f}.png'):
        """Initialise a frame object.

        Inputs: fname - filename of a piv velocity text file
                keys - a dictionary of {k: v} where k is a header
                       from the data file and v is the column it
                       corresponds to.
                quiver_format - the format of quiver file output
        """
        self.fname = fname
        self.quiver_format = quiver_format
        self.content_start = self.content_line + 2
        self.header_start = self.header_line + 1
        if stereo is False:
            self.columns = {'x': 0,
                            'z': 1,
                            'u': 6,
                            'w': 7}

        if stereo is True:
            self.columns = {'x': 2,
                            'z': 3,
                            'u': 4,
                            'v': 6,
                            'w': 5}
        for k in self.columns:
            setattr(self, k, self.data[k])
        # array of times to match dimension of other data
        self.t = np.ones(self.x.shape) * float(self.header['TimeStamp'])

    def find_line(self, string):
        """Find the line on which the string occurs
        and return it."""
        with open(self.fname) as f:
            for line_number, line in enumerate(f.readlines()):
                if string in line:
                    return line_number

    @property
    def header_line(self):
        """Find the line on which the header string occurs
        and return it."""
        header_string = ">>*HEADER*<<"
        return self.find_line(header_string)

    @property
    def content_line(self):
        """Find the line on which the header string occurs
        and return it."""
        content_string = ">>*DATA*<<"
        return self.find_line(content_string)

    @property
    def header(self):
        """Pull header from velocity file and return as dictionary."""
        with open(self.fname) as f:
            content = f.read().splitlines()
            head = content[self.header_start: self.content_start]
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
        D = np.genfromtxt(self.fname,
                          skip_header=self.content_start,
                          delimiter=delimiter)
        shape = self.shape

        # extract from given columns and reshape to sensible
        data = {k: D[:, self.columns[k]].reshape(shape) for k in self.columns}
        return data

    def make_quiver_plot(self, quiver_dir=''):
        """Make a quiver plot of the frame data."""
        fig = plt.figure()
        ax = plt.axes(xlim=(10, 80), ylim=(0, 50))
        ax.quiver(self.u, self.w, scale=200)
        quiver_name = self.quiver_name
        quiver_path = os.path.join(quiver_dir, quiver_name)
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
    frame = SingleLayerFrame(fname=fname, stereo=args['stereo'])
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
    frame = SingleLayerFrame(fname, stereo=args['stereo'])
    frame.make_quiver_plot(quiver_dir=quiver_dir)
    pbar.update()
    queue.put(1)


class SingleLayerRun(object):
    """Represents a PIV experiment. Written for a gravity current
    generated by lock release impinging on a homogeneous ambient, but
    perhaps can be made more general.
    """
    def __init__(self, index='test', data_dir='data', rex='*',
                 parallel=True, limits=None, x_lims=(0, -1),
                 stereo=False, caching=True, cache_reload=False,
                 cache_dir=None):
        """Initialise a run.

        Inputs: data_dir - directory containing the velocity files
                index    - the hash code of the run (some unique id
                           in the filenames)
                rex      - some other string found in the filenames
                parallel - whether to use multiprocessing (default True)
                limits   - (start, finish) frame indices to use in selecting
                           the list of files. Default None is to use all
                           the files.
                x_lims   - (first, last) indices of horizontal region to use
                           in each frame. e.g. (10, -10): exclude 10 cells at
                           each side.
                stereo   - boolean, is it a Stereo PIV run or not?
                caching  - boolean, enable caching?
                cache_reload - boolean, reload the data anyway?
                cache_dir - where is the cache file found?
        """
        self.index = index

        # default cache_dir
        if not cache_dir:
            cache_dir = os.path.join(data_dir, 'cache')
        self.cache_dir = cache_dir

        cache_fname = '{index}.run_object'.format(index=self.index)
        self.cache_path = os.path.join(self.cache_dir, cache_fname)

        self.data_dir = data_dir
        f_re = "*{index}{rex}".format(index=index, rex=rex)
        self.allfiles = sorted(glob.glob(os.path.join(data_dir, 'data', f_re)))

        if limits:
            first, last = limits
            self.files = self.allfiles[first:last]
        else:
            self.files = self.allfiles

        self.nfiles = len(self.files)
        self.quiver_dir = os.path.join(data_dir, 'quiver')
        self.quiver_format = 'quiver_{f}.png'
        self.parallel = parallel
        self.x_lims = x_lims
        self.stereo = stereo

        # name for frame storage attribute
        self.lazy_frames = '_lazy_frames'
        self.caching = caching
        # delete the cache file if we are reloading
        self.cache_reload = cache_reload
        if self.cache_reload and os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        # if caching enabled and the cache file exists, load it
        if self.caching and os.path.exists(self.cache_path):
            self.load()

    @property
    def frames(self):
        """List of frame objects corresponding to the input files.

        Implements lazy property evaluation in that calling self.frames
        will only perform computation once, storing the result in a
        hidden attribute (self.lazy_frames, below).
        """
        # if the storage attribute exists and caching is enabled,
        # just return the storage attribute
        if hasattr(self, self.lazy_frames):
            return getattr(self, self.lazy_frames)

        if self.parallel:
            frames = self.get_frames_parallel()
        elif not self.parallel:
            frames = self.get_frames_serial()

        # save frames to storage attribute (lazy property evaluation)
        setattr(self, self.lazy_frames, frames)

        if self.caching:
            self.save()

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
            frame = SingleLayerFrame(fname=fname, stereo=self.stereo)
            frames.append(frame)
            pbar.update(i)

        pbar.finish()
        return frames

    def get_frames_parallel(self, processors=20):
        """Get the frames with multiprocessing.
        """
        args = [dict(fname=f, stereo=self.stereo) for f in self.files]
        frames = parallel_process(instantiateFrame, args, processors)
        if type(frames) is not list:
            raise UserWarning('frames is not list!')
        # order based on filename
        sorted_frames = sorted(frames, key=lambda f: f.fname)
        return sorted_frames

    def make_quivers(self):
        """Take the text files for this run and generate quiver plots
        in parallel using multiprocessing.
        """
        print("Generating {n} quiver plots".format(n=len(self.files)))
        makedirs_p(self.quiver_dir)
        quiver_kwargs = {'quiver_dir': self.quiver_dir,
                         'stereo':     self.stereo}
        arglist = [dict(quiver_kwargs, fname=f) for f in self.files]
        parallel_process(gen_quiver_plot, arglist)

    def save(self):
        """Save run object to disk."""
        path = self.cache_path
        makedirs_p(os.path.dirname(path))
        with open(path, 'wb') as output:
            pickle.dump(self.__dict__, output, protocol=2)

    def load(self):
        """Load saved instance data from pickle file.

        Useful to avoid re-extracting the velocity data
        repeatedly.

        Can't directly assign to self: have to go through the
        dicts, see [how-to-pickle-yourself][SO]

        [SO]: http://stackoverflow.com/q/2709800/how-to-pickle-yourself
        """
        with open(self.cache_path, 'rb') as infile:
            tmp_dict = pickle.load(infile)
            self.__dict__.update(tmp_dict)

    def toggle_cache(self, what=None):
        """Toggle the state of run caching by changing the
        value of self.caching.

        Called with no arguments, reverses value of self.caching.

        Called with boolean argument, sets self.caching to that
        value.
        """
        if what is None:
            self.caching = not(self.caching)
            return
        else:
            if type(what) is not bool:
                raise UserWarning('cache value must be boolean')
                return
            else:
                self.caching = what
                return

    def reload(self):
        """Force reload of frames."""
        if self.parallel:
            frames = self.get_frames_parallel()
        elif not self.parallel:
            frames = self.get_frames_serial()
        setattr(self, self.lazy_frames, frames)
        if self.caching:
            self.save()


class SingleLayer2dRun(SingleLayerRun):
    """Represents a 2D PIV experiment. Written for a gravity current
    generated by lock release impinging on a homogeneous ambient, but
    perhaps can be made more general.
    """
    def __init__(self, **kwargs):
        SingleLayerRun.__init__(self, stereo=False, **kwargs)

    @property
    def T(self):
        return np.dstack(f.t for f in self.frames)

    @property
    def X(self):
        return np.dstack(f.x for f in self.frames)

    @property
    def Z(self):
        return np.dstack(f.z for f in self.frames)

    @property
    def U(self):
        return np.dstack(f.u for f in self.frames)

    @property
    def W(self):
        return np.dstack(f.w for f in self.frames)


class SingleLayer3dRun(SingleLayerRun):
    """Represents a 3D PIV experiment. Written for a gravity current
    generated by lock release impinging on a homogeneous ambient, but
    perhaps can be made more general.
    """
    def __init__(self, **kwargs):
        SingleLayerRun.__init__(self, stereo=True, **kwargs)

    @property
    def T(self):
        return np.dstack(f.t for f in self.frames)

    @property
    def X(self):
        return np.dstack(f.x for f in self.frames)

    @property
    def Z(self):
        return np.dstack(f.z for f in self.frames)

    @property
    def U(self):
        return np.dstack(f.u for f in self.frames)

    @property
    def V(self):
        return np.dstack(f.v for f in self.frames)

    @property
    def W(self):
        return np.dstack(f.w for f in self.frames)
