import os
import glob
from collections import OrderedDict

if 'DISPLAY' not in os.environ:
    import matplotlib as mpl
    mpl.use('Agg')

import numpy as np
import h5py
import scipy.ndimage as ndi

from util import makedirs_p
from util import parallel_process
from util import parallel_stub

from inpainting import Inpainter


def get_timestamp(fname):
    """For a Dynamic Studio export file, find the line in the header
    that contains the timestamp and extract it.

    Returns the timestamp as a np.float32

    If no timestamp is found, raises ValueError.
    """
    with open(fname) as f:
        for i, line in enumerate(f):
            if 'TimeStamp' in line:
                timestamp = np.float32(line.split(':')[1].split('\r')[0])
                return timestamp
            elif 'CONTENT' in line:
                raise ValueError("No timestamp in header.""")


@parallel_stub
def file_to_array(fname, dtype, skip_header, delimiter=None):
    """Extract file to recarray using numpy loadtxt and
    pull out the timestamp from the file.

    Returns a tuple (t, D), where t is the timestamp (float)
    and D is the recarray from np.loadtxt.
    """
    t = get_timestamp(fname)
    D = np.loadtxt(fname,
                   dtype=dtype,
                   skiprows=skip_header,
                   delimiter=delimiter,)
    return t, D


class SingleLayerFrame(object):
    """Each SingleLayerRun is comprised of a series of frames.
    This class represents one of the frames.
    """
    def __init__(self, fname, columns=None, delimiter=None):
        """Initialise a frame object.

        Inputs: fname - filename of a piv velocity text file
                columns - np.dtype, with the names and types of the
                          fields in the text file
                delimiter - data delimiter used in the file
        """
        self.fname = fname
        self.content_start = self.content_line + 2
        self.header_start = self.header_line + 1
        self.columns = columns
        self.delimiter = delimiter
        self.timestamp = float(self.header['TimeStamp'])

    def init_data(self):
        for v in self.columns.names:
            setattr(self, v, self.data[v])
        self.t = np.ones(self.shape) * self.timestamp

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
        gridsize = self.header['GridSize'].strip('{}')
        x = gridsize.split(', ')[0].split('=')[1]
        z = gridsize.split(', ')[1].split('=')[1]
        shape = (int(z), int(x))
        return shape

    @property
    def data(self):
        """Extract data from a PIV velocity text file.

        N.B. Here I've used the convention (u, w) for (streamwise,
        vertical) velocity, in contrast to the files which use (u, v).
        Similarly for (x, z) rather than (x, y).

        This is to be consistent with meteorological convention.
        """
        if hasattr(self, '_data'):
            return self._data

        if not self.delimiter:
            # force determine delimiter
            if self.header['FileID'] == 'DSExport.CSV':
                self.delimiter = ','
            elif self.header['FileID'] == 'DSExport.TAB':
                self.delimiter = None
        # extract data
        # TODO: dtypes
        D = np.genfromtxt(self.fname,
                          dtype=self.columns,
                          skip_header=self.content_start,
                          delimiter=self.delimiter,)

        self._data = D.reshape(self.shape)

        return self._data


class H5Cache(object):
    """Base class for loading from a HDF5 cache."""
    def __init__(self, cache_path=None):
        self.cache_path = cache_path
        if self.cache_path:
            self.init_cache(self.cache_path)

    @property
    def valid_cache_exists(self):
        if not hasattr(self, 'cache_path') or not self.cache_path:
            return False
        else:
            return h5py.is_hdf5(self.cache_path)

    def init_cache(self, cache_path):
        """Determine the cache_path and load if exists."""
        if os.path.isdir(cache_path):
            cache_fname = '{}.hdf5'.format(self.pattern)
            self.cache_path = os.path.join(cache_path, cache_fname)

        elif os.path.isfile(cache_path):
            self.cache_path = cache_path

        elif not os.path.exists(cache_path):
            self.cache_path = cache_path

        else:
            raise Warning('Something went wrong')

    @property
    def vectors(self):
        """Default vectors is whatever the hdf5 file has as
        its keys and types.

        Returns a numpy dtype instance. You can get the names
        of the fields with vectors.names

        You can also set vectors to whatever you want it to be.
        """
        if hasattr(self, '_vectors'):
            return self._vectors
        else:
            h5 = h5py.File(self.cache_path, 'r')
            vectors = np.dtype([(k.encode(), v.dtype) for k, v in h5.items()])
            h5.close()
            return vectors

    @vectors.setter
    def vectors(self, value):
        self._vectors = value

    def load(self, force=False):
        """Load run arrays from disk.

        Useful to avoid re-extracting the velocity data
        repeatedly.

        If the cache_path is not valid hdf5 or does not exist,
        raise UserWarning.
        """
        if not self.valid_cache_exists:
            raise UserWarning('No valid cache file found!')

        self.h5file = h5py.File(self.cache_path, 'r')

        vector_names = self.vectors.names
        different_names = set(vector_names).difference(self.h5file.keys())

        if different_names and not force:
            raise TypeError('Vector names are different to hdf5 keys')
        elif different_names and force:
            print 'Vector names are different to hdf5 keys. Overriding...'
            vector_names = self.h5file.keys()

        for v in vector_names:
            setattr(self, v, self.h5file[v])

        setattr(self, 'attributes', dict(self.h5file.attrs))

    def load_to_memory(self):
        """Load all of the vectors to memory. Careful! This can be
        O(10GB).

        You probably don't actually need to use this method as
        h5py caches in memory.
        """
        for v in self.vectors.names:
            setattr(self, v, getattr(self, v)[...])

    def hdf5_write_prep(self, cache_path):
        """Prepare for writing a new hdf5 to cache_path. Will DESTROY
        an existing HDF5 at that path.
        """
        makedirs_p(os.path.dirname(cache_path))
        # delete if a file exists. h5py sometimes complains otherwise.
        if hasattr(self, 'h5file'):
            self.h5file.close()
        if os.path.exists(cache_path):
            os.remove(cache_path)


class SingleLayerRun(H5Cache):
    """Represents a PIV experiment. Written for a gravity current
    generated by lock release impinging on a homogeneous ambient, but
    perhaps can be made more general.
    """
    # Column names and types from the text files exported by Dynamic
    # Studio. Single camera (2d) and two camera (3d) have different
    # headers.

    # N.B use 'z, w' for vertical position, velocity and 'y, v' for
    # out of page position, velocity here (DynamicStudio uses v for
    # vertical and w for out of page)

    # Why?! Because the application is meteorology and in the
    # atmosphere it is conventional to have z as the vertical axis.

    # TODO: select 2d / 3d from within the file by looking for
    # 'Stereo PIV' string?

    columns_2d = [('ix',        np.int16),    # horizontal index
                  ('iz',        np.int16),    # vertical index
                  ('x_pix',     np.float32),  # horizontal position (pixels)
                  ('z_pix',     np.float32),  # vertical position (pixels)
                  ('x',         np.float32),  # horizontal position (mm)
                  ('z',         np.float32),  # vertical position (mm)
                  ('u_pix',     np.float32),  # horizontal velocity (pixels /s)
                  ('w_pix',     np.float32),  # vertical velocity (pixels / s)
                  ('u',         np.float32),  # horizontal velocity (m / s)
                  ('w',         np.float32),  # vertical velocity (m / s)
                  ('magnitude', np.float32),  # length (m / s)
                  ('status',    np.int16),    # status code
                  ]

    columns_3d = [('ix',        np.int16),    # horizontal index
                  ('iz',        np.int16),    # vertical index
                  ('x',         np.float32),  # horizontal position (mm)
                  ('z',         np.float32),  # vertical position (mm)
                  ('u',         np.float32),  # horizontal velocity (m / s)
                  ('w',         np.float32),  # vertical velocity (m / s)
                  ('v',         np.float32),  # perpendicular velocity (m / s)
                  ('magnitude', np.float32),  # length (m / s)
                  ('status',    np.int16),    # status code
                  ]

    def __init__(self, data_dir='./', pattern='*', rex=None,
                 limits=None, stereo=True, cache_path=None):
        """Initialise a run.

        Inputs: data_dir - directory containing the velocity files
                pattern  - the hash code of the run (some unique id
                           in the filenames)
                rex      - some optional expression to match after pattern
                           in the filenames
                limits   - (start, finish) frame indices to use in selecting
                           the list of files. Default None is to use all
                           the files.
                stereo   - boolean, is it a Stereo PIV run or not?
                cache_path - string, where the hdf5 storage file is going
                             to be put. If this is a directory, the file
                             path will be directory/pattern.hdf5

        For a new run, this assumes that you have a collection of
        text files that are the result of an export from Dynamic
        Studio. The filenames of these files match a regular
        expression '*{pattern}*{rex}*' and are contained in
        `data_dir`.

            r = SingleLayerRun()

        After initialising the run, we can import the run data to
        HDF5. This is a binary storage format that is optimised for
        lookup time. A location `cache_path` needs to be specified -
        this can also be done at initialisation.

            r.import_to_hdf5(cache_path='/path/to/cache.hdf5')

        This import will put the data into RAM before writing to
        disk.  You might use 20GB of RAM! For a large number of
        files it is much quicker to do it this way than to write
        sequentially. The import will use multiprocessing to
        populate the RAM as quickly as possible.

        The Python HDF5 library, h5py, allows us to interact with
        hdf5 files as if they were numpy arrays, only reading from
        disk when necessary. This is very useful for multi GB
        experimental data.

        If a run is initialised and a valid hdf5 file exists at the
        `cache_path`, the cache will be loaded.

        On loading, all of the datasets in the hdf5 file become
        attributes of the run instance. For example, we can access
        the vertical velocities as `r.v`.

        N.B. `r.v` will return the view into the hdf5 file, which
        you can slice and process just like a numpy array, e.g.

            vertical_average = np.mean(r.v[10:20, :, :], axis=0)

        The run can be explicitly loaded by `r.load()`. If for some
        reason you want to load everything to memory, use
        `r.load_to_memory()`.
        """
        self.data_dir = data_dir
        self.pattern = pattern
        self.rex = rex or '*'

        self.limits = limits
        self.stereo = stereo

        if stereo is False:
            self.columns = np.dtype(self.columns_2d)
        elif stereo is True:
            self.columns = np.dtype(self.columns_3d)

        # name and type of time field (not in the original data files)
        time_type = ('t', np.float32)
        self.vectors = np.dtype(self.columns.descr + [time_type])

        self._init_frames = False
        if cache_path:
            self.init_cache(cache_path)
        if self.valid_cache_exists:
            self.load()

    def init_load_frames(self):
        """Initialisation to follow if we are not loading directly from
        the cache file."""
        f_re = "*{pattern}{rex}*".format(pattern=self.pattern, rex=self.rex)
        file_path = os.path.join(self.data_dir, f_re)
        self.allfiles = sorted(glob.glob(file_path))

        if len(self.allfiles) == 0:
            raise UserWarning('No files found in data dir')

        if self.limits:
            first, last = self.limits
            self.files = self.allfiles[first:last]
        else:
            self.files = self.allfiles

        self.nfiles = len(self.files)

        # mark that this init passed
        self._init_frames = True

    def frames(self):
        if not self._init_frames:
            self.init_load_frames()
        return (SingleLayerFrame(fname, self.columns) for fname in self.files)

    def import_to_hdf5(self, cache_path=None, processors=10, attributes=None):
        """Load files directly to hdf5 with multiprocessing.

        This can use a lot of RAM, ~20GB for a full run.

        Will overwrite any existing cache.

        Inputs:
            cache_path - specify where to save the hdf5
            processors - how many processors to use in multicore extraction
            attributes - dictionary of information to store in the .attrs
                         section of the hdf5.
        """
        if not self._init_frames:
            self.init_load_frames()

        cache_path = cache_path or self.cache_path
        self.hdf5_write_prep(cache_path)

        # extract the sorted data, timestamps and create array
        # you might be tempted to try and merge / append some
        # recarrays. This is a needless waste of time if you
        # just want to import to hdf5.
        timestamps, data = self.get_data(processors=processors)
        time = np.ones(data.shape) * timestamps

        # pre allocate the space in the h5 file
        h5file = h5py.File(cache_path, 'w')
        for field in self.vectors.names:
            h5file.create_dataset(field, data.shape, dtype=self.vectors[field])

        # write everything to the h5 file
        print "Writing {n} MB...".format(n=data.nbytes / 1000 ** 2)
        for vector in self.columns.names:
            h5file[vector][...] = data[vector]

        # write attributes if specified
        if attributes and hasattr(attributes, 'items'):
            for k, v in attributes.items():
                h5file.attrs[k] = v

        h5file['t'][...] = time
        h5file.close()

    def get_data(self, processors):
        """Extract raw data from ascii files and reshape."""
        # extract info from the first file
        f0 = SingleLayerFrame(self.files[0], columns=self.columns)
        if f0.header['FileID'] == 'DSExport.CSV':
            delimiter = ','
        elif f0.header['FileID'] == 'DSExport.TAB':
            delimiter = None

        kwarglist = [dict(skip_header=f0.content_start,
                          dtype=self.columns,
                          delimiter=delimiter,
                          fname=fname) for fname in self.files]

        # parallel_process returns an unsorted list so we need to
        # sort by time
        timestamps, data = zip(*parallel_process(file_to_array,
                                                 kwarglist=kwarglist,
                                                 processors=processors))
        data = np.dstack(data)
        timestamps = np.array(timestamps)

        sz, sx = f0.shape
        rdata = data.T.reshape((-1, sz, sx)).transpose((1, 2, 0))

        sorted_by_time = timestamps.argsort()
        sorted_data = rdata[:, :, sorted_by_time]
        sorted_timestamps = timestamps[sorted_by_time]

        return sorted_timestamps, sorted_data

    def get_timestamps(self):
        timestamps = np.array([get_timestamp(f) for f in self.files])
        return timestamps


class PreProcessor(H5Cache):
    """Apply basic pre processing to raw Dynamic Studio output to make
    it usable in analysis.

    Usage:

        r = SingleLayerRun
        pp = PreProcessor(r)
        pp.execute()
        pp.write_data(hdf5_cache_path)

    Stages:

        - transform X, Z to lock / base relative.
        - extract valid region
        - transform to front relative coordinates
        - interpolation of zero values (replace with nan?)
        - write to hdf5

    N.B. You won't be able to write anything until you
    have run `pp.execute()`.

    There are more methods on this class (for each of the
    steps in execute) but if they are run out of order you
    will probably get weird results.
    """
    # The origin of the coordinate system (centre of the
    # calibration target) is 105mm from the base of the tank
    # and 3250mm from the back of the lock.
    # Coordinates are x positive downstream (away from the lock)
    # and z positive upwards (away from the base).
    horizontal_offset = 3.250
    vertical_offset = 0.105

    # In the calibration coordinate system, the valid region
    # is a rectangle with lower left (-0.06, -0.10) and upper
    # right (0.10, 0.02).
    # TODO: do lock relative transform first and change these to
    # lock relative coords
    valid_region_xlim = (-0.075, 0.085)
    valid_region_ylim = (-0.10, 0.02)

    # the names of the attributes that an instance should have
    # after running self.execute()
    vectors = [('X',  np.float32),  # streamwise coordinates
               ('Z',  np.float32),  # vertical coordinates
               ('T',  np.float32),  # time coordinates
               ('U',  np.float32),  # streamwise velocity
               ('V',  np.float32),  # cross stream velocity
               ('W',  np.float32),  # vertical velocity
               ('Xf', np.float32),  # front relative streamwise coords
               ('Zf', np.float32),  # front relative vertical coords
               ('Tf', np.float32),  # front relative time coords
               ('Uf', np.float32),  # front relative streamwise velocity
               ('Vf', np.float32),  # front relative cross stream velocity
               ('Wf', np.float32),  # front relative vertical velocity
               ('fx', np.float32),  # front detection in space
               ('ft', np.float32),  # front detection in time
               ('front_speed',      # speed of the front in LAB coords
                      np.float32),
               ## NON DIMENSIONALISED copies of the above
               ('X_',  np.float32),  # streamwise coordinates
               ('Z_',  np.float32),  # vertical coordinates
               ('T_',  np.float32),  # time coordinates
               ('U_',  np.float32),  # streamwise velocity
               ('V_',  np.float32),  # cross stream velocity
               ('W_',  np.float32),  # vertical velocity
               ('Xf_', np.float32),  # front relative streamwise coords
               ('Zf_', np.float32),  # front relative vertical coords
               ('Tf_', np.float32),  # front relative time coords
               ('Uf_', np.float32),  # front relative streamwise velocity
               ('Vf_', np.float32),  # front relative cross stream velocity
               ('Wf_', np.float32),  # front relative vertical velocity
               ]
    vectors = np.dtype(vectors)

    def __init__(self, run):
        self.run = run
        self.run.load()
        self.has_executed = False

    def execute(self):
        """Execute the pre-processing steps in the
        right order. Can't write data until this has
        been done.
        """
        steps = ['extract_valid_region',
                 'filter_zeroes',
                 'interpolate_nan',
                 'transform_to_lock_relative',  # TODO: put this first
                 'transform_to_front_relative',
                 'non_dimensionalise']

        for step in steps:
            getattr(self, step)()

        self.has_executed = True

    def transform_to_lock_relative(self):
        """This method changes the X and Z coordinates such that
        they have their origin at the lock gate and the base of
        the tank, respectively.
        """
        self.X = self.X + self.horizontal_offset
        self.Z = self.Z + self.vertical_offset

    def extract_valid_region(self):
        """Extract the valid data region from the run and
        convert to SI units.
        """
        valid = self.compute_valid_slice()

        r = self.run

        self.X = r.x[valid] / 1000
        self.Z = r.z[valid] / 1000
        self.T = r.t[valid]
        self.U = r.u[valid]
        self.V = r.v[valid]
        self.W = r.w[valid]

    def compute_valid_slice(self):
        """Determine the slice to be used to pull out the valid
        region."""
        # find the indices that correspond to the rectangular view
        # that we are going to take into the data
        x_min, x_max = self.valid_region_xlim
        z_min, z_max = self.valid_region_ylim

        X = self.run.x[:, :, 0] / 1000
        Z = self.run.z[:, :, 0] / 1000

        valid = (X > x_min) & (X < x_max) & (Z > z_min) & (Z < z_max)
        iz, ix = np.where(valid)

        ix_min, ix_max = ix.min(), ix.max()
        iz_min, iz_max = iz.min(), iz.max()

        # valid region in x, z
        valid_slice = np.s_[iz_min: iz_max, ix_min: ix_max, :]

        return valid_slice

    def detect_front(self):
        """Detect the trajectory of the gravity current front
        in space and time.

        Takes the column average of the vertical velocity and
        searches for the first time at which this exceeds a
        threshold.
        """
        # TODO: make more general, find the peak instead
        threshold = 0.01  # m / s
        column_avg = self.W.mean(axis=0)
        exceed = column_avg > threshold
        # find first occurence of exceeding threshold
        front_it = np.argmax(exceed, axis=1)
        front_ix = np.indices(front_it.shape).squeeze()

        front_space = self.X[0][front_ix, front_it]
        front_time = self.T[0][front_ix, front_it]

        return front_space, front_time

    def fit_front(self):
        """Fit a straight line to the detected front
        and resample the front times from this straight
        line.
        """
        front_space, front_time = self.detect_front()
        front_function = np.poly1d(np.polyfit(front_space, front_time, 1))
        straight_time = front_function(front_space)
        return front_space, straight_time

    @staticmethod
    def compute_front_speed(front_space, front_time):
        """compute the speed of the front (assumed constant)"""
        front_speed = (np.diff(front_space) / np.diff(front_time)).mean()
        return front_speed

    @property
    def front_speed(self):
        front_space, front_time = self.fit_front()
        return self.compute_front_speed(front_space, front_time)

    @property
    def relative_sample_times(self):
        """Create the 3d array of front relative sample times, i.e.
        the time (in seconds) relative to the passage of the gravity
        current front in the FRONT frame.

        This is just a 1d array in the time axis, repeated over the
        z and x axes.
        """
        # sampling inverval in time (dt)
        dt = np.diff(self.T[0, 0, :2])

        # start and end times (s) relative to front passage
        # TODO: move somewhere higher level
        pre_front = -5
        post_front = 20
        relative_sample_times = np.arange(pre_front, post_front, dt)

        # extend over x and z
        sz, sx, = self.T.shape[:2]
        relative_sample_times = np.tile(relative_sample_times, (sz, sx, 1))

        return relative_sample_times

    def compute_front_relative_transform_coords(self, fit='1d'):
        """Calculate the index coordinates needed to transform the
        data from the LAB frame to the FRONT frame.

        In general these coordinates can be non-integer and
        negative. The output from this function is suitable for
        using in map_coordinates.

        fit - defaults to '1d', which is to fit a straight line to the
              current and resample the time from that.

              Using None will turn off any fitting and will
              just use the raw time / space detection.

        """
        if not fit:
            front_space, front_time = self.detect_front()
        elif fit == '1d':
            front_space, front_time = self.fit_front()

        self.fx = front_space
        self.ft = front_time

        # compute the times at which we need to sample the original
        # data to get front relative data by adding the time of
        # front passage* onto the relative sampling times
        # *(as a function of x)
        rtf = front_time[None, ..., None] + self.relative_sample_times

        # now we transform real coordinates to index coordinates.
        # You might want to skip this step and just compute the
        # index coordinates straight out. The reason not to do this
        # is that the coordinates can be negative and non-integer.
        # map_coordinates is used as a fancy indexer for these
        # coordinates.

        # get the real start time of the data and the
        # sampling distance in time (dt)
        rt = self.T[0, 0, :]
        dt = rt[1] - rt[0]

        # grid coordinates of the sampling times
        # (has to be relative to what the time is at
        # the start of the data).
        t_coords = (rtf - rt[0]) / dt

        # z and x coords are the same as before
        # Actually, the x_coords should be created by extending
        # front_space - it works here because front_space is the
        # same as self.X[0, :, 0]
        z_coords, x_coords = np.indices(t_coords.shape)[:2]

        # required shape of the coordinates array is
        # (3, rz.size, rx.size, rt.size)
        coords = np.concatenate((z_coords[None],
                                 x_coords[None],
                                 t_coords[None]), axis=0)
        return coords

    def transform_to_front_relative(self):
        """Transform the data into coordinates relative to the
        position of the gravity current front, i.e. from the LAB
        frame to the FRONT frame.

        The LAB frame is the frame of reference in which the data
        were originally acquired, with velocities relative to the
        lab rest frame, times relative to the experiment start and
        space relative to the calibration target.

        The FRONT frame is the frame of reference in which the
        gravity current front is at rest, with velocities relative
        to the front, times relative to the time of front passage
        and space as in the LAB frame.

        Implementation takes advantage of regular rectangular data
        and uses map_coordinates.
        """
        coords = self.compute_front_relative_transform_coords()

        # use order 0 because 6x as fast here (3s vs 20s) and for x
        # and z it makes no difference
        self.Xf = ndi.map_coordinates(self.X, coords, order=0, cval=np.nan)
        self.Zf = ndi.map_coordinates(self.Z, coords, order=0, cval=np.nan)

        # these are the skewed original times (i.e. LAB frame)
        self.Tfs = ndi.map_coordinates(self.T, coords, order=3, cval=np.nan)
        # these are the times relative to front passage (i.e. FRONT frame)
        self.Tf = self.relative_sample_times

        # the streamwise component is in the FRONT frame
        fs = self.front_speed
        self.Uf = ndi.map_coordinates(self.U, coords, cval=np.nan) - fs
        # cross-stream, vertical components
        self.Vf = ndi.map_coordinates(self.V, coords, cval=np.nan)
        self.Wf = ndi.map_coordinates(self.W, coords, cval=np.nan)

        # N.B. there is an assumption here that r.t, r.z and r.x are
        # 3d arrays. They are redundant in that they repeat over 2 of
        # their axes (r.z, r.x, r.t = np.meshgrid(z, x, t, indexing='ij'))

    def transform_to_front_relative_space(self, fit='1d'):
        """Transform into a spatial front relative system."""
        # TODO: abstract front transform and combine this with
        # temporal transform
        if not fit:
            front_space, front_time = self.detect_front()
        elif fit == '1d':
            front_space, front_time = self.fit_front()

        rx = self.X[0, :, 0]
        dx = rx[1] - rx[0]

        # TODO: specify x0, x1 as arguments
        x0 = -0.2
        x1 = 0.2
        relative_samples = np.arange(x0, x1, dx)
        sz, sx = self.T.shape[:2]
        # size of sampling axis
        ss = relative_samples.size
        # FIXME: below is wrong
        # might be working with the above new axes
        xrelative_samples = np.tile(relative_samples, (sz, sx, 1))

        rxf = front_space[None, ..., None] + xrelative_samples

        # grid coordinates of the sampling times
        # (has to be relative to what the time is at
        # the start of the data).
        x_coords = (rxf - rx[0]) / dx

        # z coords are the same as before
        z_coords = np.indices(x_coords.shape)[0]

        # t coords are front_time extended over the sampling volume
        # plus whatever else you want to be able to see
        rt = self.T[0, 0, :]
        xfront_time = np.tile(front_time[None, ..., None], (sz, 1, ss))
        t_coords = (xfront_time - rt[0]) / (rt[1] - rt[0])

        # required shape of the coordinates array is
        # (3, rz.size, rx.size, rt.size)
        coords = np.concatenate((z_coords[None],
                                 x_coords[None],
                                 t_coords[None]), axis=0)

        # there are now two x dimensions - the original and the
        # sampling
        self.Xss = xrelative_samples
        self.Xfs = self.X[:, :, 0, None].repeat(ss, axis=-1)
        self.Zfs = self.Z[:, :, 0, None].repeat(ss, axis=-1)
        self.Tfs = xfront_time

        self.Ufs = ndi.map_coordinates(self.U, coords, cval=np.nan)
        self.Vfs = ndi.map_coordinates(self.V, coords, cval=np.nan)
        self.Wfs = ndi.map_coordinates(self.W, coords, cval=np.nan)

    def filter_zeroes(self):
        """Set all velocities that are identically zero to be nan."""
        self.U[self.U == 0] = np.nan
        self.V[self.V == 0] = np.nan
        self.W[self.W == 0] = np.nan

    def filter_anomalies(self):
        """Find anomalous data and set to nan.

        You should run this either before filter_zeroes or after
        interpolate_nan, or you get lots more nans.

        After running this you should run interpolate_nan.
        """
        # TODO: write me!
        smoothed = ndi.uniform_filter(self.U, size=3)
        thresh = 0.05  # TODO: set more generally
        bad = np.abs(self.U - smoothed) > thresh
        self.U[bad] = np.nan

    def interpolate_nan(self, sub_region=None, scale='auto'):
        """The raw data contains regions with velocity identical
        to zero. These are non physical and can be removed by
        interpolation.
        """
        if scale == 'auto':
            scale = self.front_speed

        inpainter = Inpainter(self, sub_region=sub_region, scale=scale)
        inpainter.paint(processors=20)

    def non_dimensionalise(self):
        """Take the original, dimensional run data, divide by length
        / time scales and resample to get non-dimensionalised data
        on a regular grid.

        NB. This method is gregarious with the data. It will take
        all of the run data and non-dim. It will not restrict to a
        particular regular grid, so you won't be able to stack
        multiple runs directly if they occupy different volumes in
        non-dimensional space (which they will if they have
        different parameters).
        """
        p = self.run.attributes

        # determine the scaling factors
        L = p['L']  # length
        H = p['H']  # height

        # acceleration (reduced gravity)
        g_ = 9.81 * (p['rho_lock'] - p['rho_ambient']) / p['rho_ambient']

        U = (g_ * H) ** .5  # speed
        T = H / U  # time

        # Sampling intervals in dim space. These shouldn't vary run
        # to run but they might. Maybe add this as a sanity check
        # for each run?
        dz = np.diff(self.Z[:2, 0, 0])[0]
        dx = np.diff(self.X[0, :2, 0])[0]
        dt = np.diff(self.T[0, 0, :2])[0]

        # Sampling intervals in non dim space. These are set here to
        # be constant across all runs. They were roughly determined
        # by doubling the non-dimensionalised intervals of the
        # fastest / tallest run (so that we don't try and oversample
        # anything).
        # TODO: define this elsewhere (class attribute?)
        dx_ = 0.01
        dz_ = 0.012
        dt_ = 0.015

        # as well as scaling the quantities, we have to scale the
        # sampling interval. The dimensional non dim interval is
        # dt_ * T; the dimensional interval is dt.
        zoom_factor = (dz / (H * dz_),
                       dx / (L * dx_),
                       dt / (T * dt_))

        zoom_kwargs = {'zoom':  zoom_factor,
                       'order': 1,          # spline interpolation.
                       'mode': 'constant',  # points outside the boundaries
                       'cval': np.nan,      # are set to np.nan
                       }

        self.Z_ = ndi.zoom(self.Z[:] / H, **zoom_kwargs)
        self.X_ = ndi.zoom(self.X[:] / L, **zoom_kwargs)
        self.T_ = ndi.zoom(self.T[:] / T, **zoom_kwargs)

        self.U_ = ndi.zoom(self.U[:] / U, **zoom_kwargs)
        self.V_ = ndi.zoom(self.V[:] / U, **zoom_kwargs)
        self.W_ = ndi.zoom(self.W[:] / U, **zoom_kwargs)

        self.Zf_ = ndi.zoom(self.Zf[:] / H, **zoom_kwargs)
        self.Xf_ = ndi.zoom(self.Xf[:] / L, **zoom_kwargs)
        self.Tf_ = ndi.zoom(self.Tf[:] / T, **zoom_kwargs)

        self.Uf_ = ndi.zoom(self.Uf[:] / U, **zoom_kwargs)
        self.Vf_ = ndi.zoom(self.Vf[:] / U, **zoom_kwargs)
        self.Wf_ = ndi.zoom(self.Wf[:] / U, **zoom_kwargs)

    def write_data(self, path):
        """Save everything to a new hdf5."""
        if not self.has_executed:
            print "Data has not been processed! Not writing."
            return

        self.hdf5_write_prep(path)
        h5file = h5py.File(path, 'w')

        for vector in self.vectors.names:
            data = getattr(self, vector)
            h5file.create_dataset(vector, data.shape, dtype=data.dtype)
            h5file[vector][...] = data

        for k, v in self.run.attributes.items():
            h5file.attrs[k] = v

        h5file.close()


class ProcessedRun(H5Cache):
    """Wrapper around a run that has had its data quality controlled."""
    def __init__(self, cache_path=None, forced_load=False):
        """Initialise a processed run.

        cache_path - hdf5 to load from
        forced_load - load hdf5 even its keys aren't the same as vectors
        """
        # specify vectors as those produced by the pre processor
        self.vectors = np.dtype(PreProcessor.vectors)
        self.cache_path = cache_path
        if self.cache_path:
            self.init_cache(self.cache_path)
            self.load(force=forced_load)


class Parameters(object):
    """For a given run index, determine the type of run
    (single layer / two layer) and load the appropriate
    parameters.

    An instance of this class is a function that returns
    a (ordered) dictionary of the run parameters.
    """
    # TODO: this path should be in init or config or somewhere else
    root = os.environ['HOME'] + '/lab/data/flume2/main_data/'
    single_layer_parameters = os.path.join(root, 'params_single_layer')
    two_layer_parameters = os.path.join(root, 'params_two_layer')

    single_layer_headers = [('run_index',        '|S10'),
                            ('H',                np.float),
                            ('D',                np.float),
                            ('L',                np.float),
                            ('rho_ambient',      np.float),
                            ('rho_lock',         np.float),
                            ('T_ambient',        np.float),
                            ('T_lock',           np.float),
                            ('n_sample_ambient', np.float),
                            ('n_sample_lock',    np.float),
                            ('T_sample_ambient', np.float),
                            ('T_sample_lock',    np.float),
                            ]

    two_layer_headers = [('run_index',      '|S10'),
                         ('H',              np.float),
                         ('D',              np.float),
                         ('L',              np.float),
                         ('h_1 / H',        np.float),
                         ('rho_upper',      np.float),
                         ('rho_lower',      np.float),
                         ('rho_lock',       np.float),
                         ('T_upper',        np.float),
                         ('T_lower',        np.float),
                         ('T_lock',         np.float),
                         ('n_sample_upper', np.float),
                         ('n_sample_lower', np.float),
                         ('n_sample_lock',  np.float),
                         ('T_sample_upper', np.float),
                         ('T_sample_lower', np.float),
                         ('T_sample_lock',  np.float),
                         ]

    def __init__(self, run_index=None):
        self.init_parameters()

    def __call__(self, run_index):
        run_type = self.determine_run_type(run_index)
        if run_type == 'single layer':
            return self.get_run_info(self.single_layer, run_index)
        elif run_type == 'two layer':
            return self.get_run_info(self.two_layer, run_index)
        else:
            return None

    def get_run_info(self, parameters, index):
        """Return the info for a given run index as an
        OrderedDict.
        """
        line = np.where(parameters['run_index'] == index)
        info = parameters[line]
        keys = info.dtype.names
        values = info[0]
        odict = OrderedDict(zip(keys, values))
        odict['run type'] = self.determine_run_type(index)
        return odict

    def determine_run_type(self, run_index):
        """Returns the run type as a string, either
        'two_layer' or 'single_layer'.
        """
        if run_index in self.single_layer['run_index']:
            return 'single layer'
        elif run_index in self.two_layer['run_index']:
            return 'two layer'
        else:
            return None

    def init_parameters(self):
        """Load the parameters files."""
        self.single_layer = self.load_parameters(self.single_layer_parameters,
                                                 self.single_layer_headers)
        self.two_layer = self.load_parameters(self.two_layer_parameters,
                                              self.two_layer_headers)

    @staticmethod
    def load_parameters(file, headers):
        return np.loadtxt(file, dtype=headers, skiprows=2)
