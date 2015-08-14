import numpy as np
import scipy.ndimage as ndi


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


def detect_front(run):
    """Detect the trajectory of the gravity current front
    in space and time.

    Takes the column average of the vertical velocity and
    searches for the first time at which this exceeds a
    threshold.
    """
    # TODO: make more general, find the peak instead
    threshold = 0.01  # m / s
    column_avg = run.W[:].mean(axis=0)
    exceed = column_avg > threshold
    # find first occurence of exceeding threshold
    front_it = np.argmax(exceed, axis=1)
    front_ix = np.indices(front_it.shape).squeeze()

    front_space = run.X[0][front_ix, front_it]
    front_time = run.T[0][front_ix, front_it]

    return front_space, front_time


def front_speed(run):
    front_space, front_time = fit_front(*detect_front(run))
    return compute_front_speed(front_space, front_time)


class FrontTransformer(object):
    def __init__(self, front_speed, order=0, cval=np.nan, dx=1, dt=1):
        self.front_speed = front_speed
        self.index_speed = front_speed * (dt / dx)

        self.order = order
        self.cval = cval

    @property
    def front_transform(self):
        s = 1 / self.index_speed
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, s, 1]])

    @property
    def lab_transform(self):
        s = 1 / self.index_speed
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, -s, 1]])

    def to_front(self, velocity):
        return ndi.affine_transform(velocity,
                                    self.front_transform,
                                    order=self.order,
                                    cval=self.cval)

    def to_lab(self, velocity):
        return ndi.affine_transform(velocity,
                                    self.lab_transform,
                                    order=self.order,
                                    cval=self.cval)
