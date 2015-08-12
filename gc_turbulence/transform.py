import numpy as np
import scipy.ndimage as ndi


class FrontTransformer(object):
    """Create a function that transforms data between coordinate systems."""
    def __init__(self, run, pre_front=-5, post_front=20):
        # start and end times (s) relative to front passage
        # TODO: move somewhere higher level
        self.pre_front = pre_front
        self.post_front = post_front

        self.dt = run.dt

        self.front_time = run.ft[:][None, ..., None]

        self.T_shape = run.T.shape
        self.Tmin = np.min(run.T)
        self.Tmax = np.max(run.T)

        self.lab_sample_times = run.T[...]

    @property
    def front_relative_sample_times(self):
        """Create the 3d array of front relative sample times, i.e.
        the time (in seconds) relative to the passage of the gravity
        current front in the FRONT frame.

        This is just a 1d array in the time axis, repeated over the
        z and x axes.
        """
        relative_sample_times = np.arange(self.pre_front,
                                          self.post_front,
                                          self.dt)
        # extend over x and z
        sz, sx, _ = self.T_shape
        relative_sample_times = np.tile(relative_sample_times, (sz, sx, 1))

        return relative_sample_times

    @property
    def all_front_relative_sample_times(self):
        """Front relative times using the full extent of the data."""
        pre_front = self.Tmin - np.max(self.front_time)
        post_front = self.Tmax - np.min(self.front_time)

        relative_sample_times = np.arange(pre_front, post_front, self.dt)
        # extend over x and z
        sz, sx, _ = self.T_shape
        return np.tile(relative_sample_times, (sz, sx, 1))

    @staticmethod
    def transform(vector, coords, order=1, cval=np.nan):
        """vector - array of data to sample from
        coords - grid coordinates to sample at
        order - spline interpolation order. Values greater than 1
                will cause problems with data containing nans.
        cval - value to pad output array with outside of box formed
               by coords
        """
        return ndi.map_coordinates(vector, coords, cval=cval, order=order)

    def to_reduced_front(self, array, **kwargs):
        """Transform array of data from the LAB frame to the FRONT frame."""
        return self.transform(array, self.reduced_front_coords, **kwargs)

    def to_front(self, array, **kwargs):
        """Transform array of data from the LAB frame to the FRONT frame."""
        return self.transform(array, self.front_coords, **kwargs)

    def to_lab(self, array, **kwargs):
        """Transform array of data from the FRONT frame to the LAB frame."""
        return self.transform(array, self.lab_coords, **kwargs)

    @staticmethod
    def compute_grid_coordinates(times, t0, dt):
        """Transform real coordinates to index coordinates.

        Assuming that the array of times has been produced by
        skewing in (x, t), compute the grid coordinates that would
        let us index the times from the original array.

        In general these coordinates can be non-integer and
        negative. The output from this function is suitable for
        using in map_coordinates.
        """
        # grid coordinates of the sampling times
        # (has to be relative to what the time is at
        # the start of the data).
        t_coords = (times - t0) / dt
        # Strictly, the x_coords should be created by extending
        # front_space - it works here because front_space is the
        # same as run.X[0, :, 0]
        z_coords, x_coords = np.indices(t_coords.shape)[:2]
        # required shape of the coordinates array is
        # (3, rz.size, rx.size, rt.size)
        coords = np.concatenate((z_coords[None],
                                 x_coords[None],
                                 t_coords[None]), axis=0)
        return coords

    @property
    def lab_coords(self):
        """Coordinates for transforming from FRONT to LAB, i.e. the
        coordinates with which you index front relative data to get
        lab relative data."""
        lab_times = self.lab_sample_times - self.front_time
        t0 = lab_times.min()
        return self.compute_grid_coordinates(lab_times, t0, self.dt)

    @property
    def front_coords(self):
        """Coordinates for transforming from LAB to FRONT, i.e. the
        coordinates with which you index lab relative data to get
        front relative data."""
        front_times = self.all_front_relative_sample_times + self.front_time
        return self.compute_grid_coordinates(front_times, self.Tmin, self.dt)

    @property
    def reduced_front_coords(self):
        """Coordinates for transforming from LAB to FRONT, i.e. the
        coordinates with which you index lab relative data to get front
        relative data.

        This coordinate set transforms to a reduced set of
        coordinates that only include data a set time before and
        after the front.

        If you want to transform between front and lab multiple
        times you should use `front_coords`.
        """
        # compute the times at which we need to sample the original
        # data to get front relative data by adding the time of
        # front passage* onto the relative sampling times
        # *(as a function of x)
        front_times = self.front_relative_sample_times + self.front_time
        return self.compute_grid_coordinates(front_times, self.Tmin, self.dt)


def detect_front(run):
    """Detect the trajectory of the gravity current front
    in space and time.

    Takes the column average of the vertical velocity and
    searches for the first time at which this exceeds a
    threshold.
    """
    # TODO: make more general, find the peak instead
    threshold = 0.01  # m / s
    column_avg = run.W.mean(axis=0)
    exceed = column_avg > threshold
    # find first occurence of exceeding threshold
    front_it = np.argmax(exceed, axis=1)
    front_ix = np.indices(front_it.shape).squeeze()

    front_space = run.X[0][front_ix, front_it]
    front_time = run.T[0][front_ix, front_it]

    return front_space, front_time


def fit_front(front_space, front_time):
    """Fit a straight line to the detected front
    and resample the front times from this straight
    line.
    """
    front_function = np.poly1d(np.polyfit(front_space, front_time, 1))
    straight_time = front_function(front_space)
    return front_space, straight_time


def compute_front_speed(front_space, front_time):
    """compute the speed of the front (assumed constant)"""
    front_speed = (np.diff(front_space) / np.diff(front_time)).mean()
    return front_speed


def front_speed(run):
    front_space, front_time = fit_front(*detect_front(run))
    return compute_front_speed(front_space, front_time)


def transform_to_front_relative(self, fit='1d'):
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
    if not fit:
        front_space, front_time = self.detect_front()
    elif fit == '1d':
        front_space, front_time = self.fit_front()

    self.fx = front_space
    self.ft = front_time

    transform = FrontTransformer(self, pre_front=-5, post_front=20)

    # use order 0 (nearest neighbour) because 6x as fast here
    # (3s vs 20s) and for x and z it makes no difference
    space_order = 0

    # using (spline interpoplation) order greater than 1 leads
    # to nans poisoning the data from the edges (we have already
    # removed all nans from the interior of the data in a
    # previous step). Explicitly dealing with the boundaries is
    # the proper way to get around this, but is overkill here.

    # Use order 1 (linear interpolation) for time and speed
    time_order = 1
    speed_order = 1

    self.Xf = transform.to_reduced_front(self.X, order=space_order)
    self.Zf = transform.to_reduced_front(self.Z, order=space_order)

    # these are the skewed original times (i.e. LAB frame)
    self.Tfs = transform.to_reduced_front(self.T, order=time_order)
    # these are the times relative to front passage (i.e. FRONT frame)
    self.Tf = transform.front_relative_sample_times
    # equally, self.Tf = transform.to_front(self.T) - front_time

    # the streamwise component is in the FRONT frame
    self.Uf = transform.to_reduced_front(self.U, order=speed_order) \
                - self.front_speed
    # cross-stream, vertical components
    self.Vf = transform.to_reduced_front(self.V, order=speed_order)
    self.Wf = transform.to_reduced_front(self.W, order=speed_order)

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
