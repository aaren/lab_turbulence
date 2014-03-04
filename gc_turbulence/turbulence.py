import os
import glob

import numpy as np
import h5py
import scipy.ndimage as ndi
from matplotlib.pyplot import imread

if 'DISPLAY' not in os.environ:
    import matplotlib as mpl
    mpl.use('Agg')

from util import makedirs_p
from util import parallel_process
from util import parallel_stub


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
    def __init__(self, fname, columns, delimiter=None):
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


class SingleLayerRun(object):
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

    def import_to_hdf5(self, cache_path=None, processors=10):
        """Load files directly to hdf5 with multiprocessing.

        This can use a lot of RAM, ~20GB for a full run.

        Will overwrite any existing cache.
        """
        self.init_load_frames()

        if not (cache_path or self.cache_path):
            return "No cache file specified!"
        cache_path = cache_path or self.cache_path

        makedirs_p(os.path.dirname(cache_path))
        # delete if a file exists. h5py sometimes complains otherwise.
        print cache_path
        if hasattr(self, 'h5file'):
            self.h5file.close()
        if os.path.exists(cache_path):
            os.remove(cache_path)

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

    def load(self):
        """Load run arrays from disk.

        Useful to avoid re-extracting the velocity data
        repeatedly.

        If the cache_path is not valid hdf5 or does not exist,
        raise UserWarning.
        """
        if not self.valid_cache_exists:
            raise UserWarning('No valid cache file found!')

        if hasattr(self, 'h5file'):
            try:
                self.h5file.close()
            except ValueError:
                # case file already closed
                pass

        self.h5file = h5py.File(self.cache_path, 'r')
        for v in self.vectors.names:
            setattr(self, v, self.h5file[v])

    def load_to_memory(self):
        """Load all of the vectors to memory. Careful! This can be
        O(10GB).

        You probably don't actually need to use this method as
        h5py caches in memory.
        """
        for v in self.vectors.names:
            setattr(self, v, getattr(self, v)[...])


class Validator(object):
    # invert the images on load
    s = np.s_[::-1, :]
    mask_image_1 = imread('/home/eeaol/lab/data/flume2/main_data/'
                          'image_limits/Define Mask - Reject(3nvpvi2w)/'
                          'image_mask.3nvpvi2w.000000.bmp')[s].astype(np.bool)
    mask_image_2 = imread('/home/eeaol/lab/data/flume2/main_data/'
                          'image_limits/Define Mask - Reject(3nvpz1v5)/'
                          'image_mask.3nvpz1v5.000000.bmp')[s].astype(np.bool)

    # scale and offset for converting the x, y indices to real
    # coordinates
    scale_factor_1 = ((0.250052, 0.10244), (0.175165, 0.112465))
    scale_factor_2 = ((0.235529, 0.11395), (0.167744, 0.114789))

    def __init__(self, run):
        self.run = run

    @staticmethod
    def find_index_edges_of_domain(mask_image):
        """mask_image is a boolean array of validity over the
        measurement domain. This is True in the middle (where the
        data is valid) and False around the edges.

        This function finds the indices of the edges of the valid domain.
        """
        dx, dy = np.gradient(mask_image)
        is_edge = np.where(np.logical_or(dx, dy))
        return is_edge

    @staticmethod
    def calculate_real_coords(mask_image, (scale_x, offset_x),
                              (scale_y, offset_y)):
        """Convert the indices of mask_image to real coordinates,
        using a given scale and offset.
        """
        iy, ix = np.indices(mask_image.shape)
        sy, sx = mask_image.shape

        ry = scale_y * iy / sy - offset_y
        rx = scale_x * ix / sx - offset_x

        return ry, rx

    @staticmethod
    def find_edges_of_domain(mask_image, convx, convy):
        """Calculate the edges of the valid domain in the mask_image
        using real coordinates, calculated using the (scale, offset)
        pairs given by convx and convy.

        Returns a (N, 2) array of the edge points.
        """
        ry, rx = Validator.calculate_real_coords(mask_image, convx, convy)
        is_edge = Validator.find_index_edges_of_domain(mask_image)
        edge_x = rx[is_edge]
        edge_y = ry[is_edge]
        return np.array((edge_x, edge_y)).T

    @staticmethod
    def ccworder(A):
        A = A - np.mean(A, 1)[:, None]
        return np.argsort(np.arctan2(A[1, :], A[0, :]))

    @staticmethod
    def test_is_valid(edge_points):
        """Create a method that tests whether given points lie
        within the edge defined by edge_points.

        Orders the points by their angle about the origin and
        uses a matplotlib Path instance."""
        ordered_edge = edge_points[Validator.ccworder(edge_points.T)]
        edge_path = mpl.path.Path(ordered_edge)
        return edge_path.contains_points

    def valid(self):
        """Return a boolean map of where the run data is valid."""
        edge_points_1 = self.find_edges_of_domain(self.mask_image_1,
                                                  *self.scale_factor_1)
        edge_points_2 = self.find_edges_of_domain(self.mask_image_2,
                                                  *self.scale_factor_2)

        test_valid_1 = self.test_is_valid(edge_points_1)
        test_valid_2 = self.test_is_valid(edge_points_2)

        flat_x = self.run.x[:].flatten()
        flat_z = self.run.z[:].flatten()

        shape = self.run.x.shape

        valid_1 = test_valid_1(np.vstack((flat_x, flat_z)).T).reshape(shape)
        valid_2 = test_valid_2(np.vstack((flat_x, flat_z)).T).reshape(shape)

        return valid_1 & valid_2


class AnalysisRun(object):
    def __init__(self, cache_path, pattern=None):
        self.r = SingleLayerRun(cache_path=cache_path, pattern=pattern)

        # TODO: non-dimensionalise!
        for v in self.r.vectors.names:
            arr = getattr(self.r, v)
            # remove the invalid edges
            # TODO: use masking?
            # FIXME: how do you create a partial view of hdf5
            # without loading all to memory?
            setattr(self, v, arr)

        # time of front passage as f(x) (index based)
        self.time_front = self.index_front_detect()

        # front velocity dx/dt
        # TODO: sort out non dimensional units
        self.front_velocity = 1 / self.time_front[1]

        self.T_width = 1500
        self.front_offset = -50

    def init_reshape(self):
        # self.uf = self.reshape_to_current_relative(self.u)
        self.wf = self.reshape_to_current_relative(self.w)
        # self.xf = self.reshape_to_current_relative(self.x)
        # self.zf = self.reshape_to_current_relative(self.z)
        # self.tf = self.reshape_to_current_relative(self.t)

    def index_front_detect(self):
        # It would be useful to detect where the front is in each
        # image, so that we can do meaningful statistics relative to
        # this.

        # look at the peak in the vertical component of velocity
        vertical_mean = np.mean(self.w, axis=0)
        # smooth in time
        smooth_vertical_mean = ndi.gaussian_filter1d(vertical_mean,
                                                     sigma=30,
                                                     axis=1)
        # Front position is defined as the maximum of the smoothed
        # vertical mean.  For each of the x indices, these are the
        # indices of the maxima in the time coordinate
        maxima_it = np.argmax(smooth_vertical_mean, axis=1)
        # corresponding x indices
        maxima_ix = np.indices(maxima_it.shape).squeeze()

        ## If you want the actual x, t values for the front passage
        ## you do this:
        ##
        ## x = self.x[0][maxima_ix, maxima_it]
        ## t = self.t[0][maxima_ix, maxima_it]

        linear_fit = np.poly1d(np.polyfit(maxima_ix, maxima_it, 1))
        # now given position x , get time of front pass t = linear_fit(x)
        return linear_fit

    def reshape_to_current_relative(self, vel):
        """Take the velocity data and transform it to the current
        relative frame.
        """
        # TODO: use interpolation instead of int(round())
        # tf is the time of front passage as f(x), i.e. supply this
        # with an argument in x and we get the corresponding time
        # reshape, taking a constant T time intervals behind front
        X = np.indices((vel.shape[1],)).squeeze()

        def extract(vel, x, front_offset, dt):
            tf = self.time_front
            front_index = int(round(tf(x)))
            t0 = front_index + front_offset
            t1 = front_index + dt + front_offset
            return vel[:, x, t0: t1]

        U_ = np.dstack(extract(vel, x, self.front_offset, self.T_width)
                       for x in X)
        # reshape this to same dimensions as before
        Uf = np.transpose(U_, (0, 2, 1))
        # TODO: does axis 2 of Uf need to be reversed?
        # reverse axis so that time axis progresses as time in the
        # evolution of the front
        # Uf = Uf[:,:,::-1]
        return Uf


class PreProcessor(object):
    """Apply basic pre processing to raw Dynamic Studio output to make
    it usable in analysis.

    Stages:

        - transform X, Z to lock / base relative.
        - extract valid region
        - transform to front relative coordinates
        - interpolation of zero values (replace with nan?)
        - write to hdf5
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
    valid_region_xlim = (-0.075, 0.085)
    valid_region_ylim = (-0.10, 0.02)

    # the names of the attributes that an instance should have
    # after running self.execute()
    vectors = ['X', 'Z', 'T',     # coordinate
               'U', 'V', 'W',     # values
               'X_', 'Z_', 'T_',  # front relative coordinates
               'U_', 'V_', 'W_']  # front relative data

    def __init__(self, run):
        self.run = run
        self.has_executed = False

    ## TODO: specify order in which these methods must be run.
    def execute(self):
        self.extract_valid_region()
        self.interpolate_zeroes()
        self.transform_to_lock_relative()
        self.transform_to_front_relative()
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
        threshold = 0.01  # m / s
        column_avg = self.W.mean(axis=0)
        exceed = column_avg > threshold
        # find first occurence of exceeding threshold
        front_it = np.argmax(exceed, axis=1)
        front_ix = np.indices(front_it.shape).squeeze()

        front_space = self.X[0][front_ix, front_it]
        front_time = self.T[0][front_ix, front_it]

        return front_space, front_time

    def transform_to_front_relative(self):
        """Transform the data into coordinates relative to the
        position of the gravity current front.

        Implementation takes advantage of regular rectangular data
        and uses map_coordinates.
        """
        front_space, front_time = self.detect_front()

        # get the real start time of the data and the
        # sampling distance in time (dt)
        rt = self.T[0, 0, :]
        dt = rt[1] - rt[0]

        ## compute the times at which to sample the *front relative*
        ## data, if it already existed as a 3d rectangular array (which it
        ## doesn't!). The other way to do it would be to compute the
        ## necessary sample indices first and then index the r.t array.
        ## You shouldn't do it that way because the coords can be negative
        ## floats and map_coordinates will handle that properly.
        #
        # start and end times (s) relative to front passage
        t0 = -5
        t1 = 20
        relative_sample_times = np.arange(t0, t1, dt)
        # extend over x and z
        sz, sx, = self.T.shape[:2]
        relative_sample_times = np.tile(relative_sample_times, (sz, sx, 1))

        ## now compute the times at which we need to sample the
        ## original data to get front relative data by adding
        ## the time of front passage* onto the relative sampling times
        ## *(as a function of x)
        rtf = front_time[None, ..., None] + relative_sample_times

        # grid coordinates of the sampling times
        # (has to be relative to what the time is at
        # the start of the data).
        t_coords = (rtf - rt[0]) / dt

        # z and x coords are the same as before
        zx_coords = np.indices(t_coords.shape)[:2]

        # required shape of the coordinates array is
        # (3, rz.size, rx.size, rt.size)
        coords = np.concatenate((zx_coords, t_coords[None]), axis=0)

        st = t_coords.shape[-1]
        self.X_ = self.X[:, :, 0, None].repeat(st, axis=-1)
        self.Z_ = self.Z[:, :, 0, None].repeat(st, axis=-1)
        self.T_ = relative_sample_times

        self.U_ = ndi.map_coordinates(self.U, coords)
        self.V_ = ndi.map_coordinates(self.V, coords)
        self.W_ = ndi.map_coordinates(self.W, coords)

        # N.B. there is an assumption here that r.t, r.z and r.x are
        # 3d arrays. They are redundant in that they repeat over 2 of
        # their axes (r.z, r.x, r.t = np.meshgrid(z, x, t, indexing='ij'))

    def interpolate_zeroes(self):
        """The raw data contains regions with velocity identical
        to zero. These are non physical and can be removed by
        interpolation.
        """
        # TODO: write me!

    def write_data(self):
        """Save everything to a new hdf5."""
        if not self.has_executed:
            print "Data has not been processed! Not writing."
            return

        raw_cache = self.run.cache_path
        raw_root, ext = os.path.splitext(raw_cache)

        pro_root = raw_root + '_processed'
        cache_path = pro_root + ext

        makedirs_p(os.path.dirname(cache_path))
        # delete if a file exists. h5py sometimes complains otherwise.
        if hasattr(self, 'h5file'):
            self.h5file.close()
        if os.path.exists(cache_path):
            os.remove(cache_path)

        h5file = h5py.File(cache_path, 'w')

        for vector in self.vectors:
            data = getattr(self, vector)
            h5file.create_dataset(vector, data.shape, dtype=data.dtype)
            h5file[vector][...] = data

        h5file.close()


class NonDimensionaliser(object):
    """Take a run and non dimensionalise it, re-mapping the data to
    a standard regular grid.
    """


class ProcessedRun(object):
    """Wrapper around a run that has had its data quality controlled."""
    # attribute names that the data will have when loaded from the
    # hdf5
    vectors = PreProcessor.vectors

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

    def load(self):
        """Load run arrays from disk.

        Useful to avoid re-extracting the velocity data
        repeatedly.

        If the cache_path is not valid hdf5 or does not exist,
        raise UserWarning.
        """
        if not self.valid_cache_exists:
            raise UserWarning('No valid cache file found!')

        if hasattr(self, 'h5file'):
            try:
                self.h5file.close()
            except ValueError:
                # case file already closed
                pass

        self.h5file = h5py.File(self.cache_path, 'r')
        for v in self.vectors:
            setattr(self, v, self.h5file[v])

    def load_to_memory(self):
        """Load all of the vectors to memory. Careful! This can be
        O(10GB).

        You probably don't actually need to use this method as
        h5py caches in memory.
        """
        for v in self.vectors.names:
            setattr(self, v, getattr(self, v)[...])
