import sys
import multiprocessing as mp

import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp

import gc_turbulence as g

import os

os.system("taskset -p 0xffffffff %d" % os.getpid())


class Inpainter(object):
    """Fills holes in gridded data (represented by nan) by
    performing linear interpolation using only the set of points
    surrounding each hole.
    """
    def __init__(self, run, sub_region=None, scale=1):
        """run has attributes representing coordinates (X, Z, T)
        and data (U, V, W).

        sub_region lets you optionally consider only a sub section
        of the data by indexing. Default is to take all of the data.

        scale is the factor to multiply the time coordinates by to
        make them comparable with the spatial coordinates in
        euclidean space. This should be a characteristic speed of
        the flow (e.g. the front velocity)
        """
        self.run = run
        self.sub = sub_region or np.s_[:, :, :]

        # factors to scale the run coordinates by (Z, X, T)
        self.scales = (('Z', 1),
                       ('X', 1),
                       ('T', scale))

        self.data_names = ('U', 'V', 'W')

        # this has to go last
        self.setup()

    def setup(self):
        """Perform the initialisation needed to interpolate over
        regions of nan.

        Linear interpolation scales with N^dimension, so we want to
        limit the number of points that we use in the interpolator
        as much as possible.

        We label all of the nan values in the data (self.invalid),
        then find the valid points that directly surround them
        (self.complete_valid_shell), as this is all we need for
        linear interpolation. We refer to the surrounding valid
        points as the valid shell.

        We separate the shells and the invalid points within into
        clusters (volumes), as it is wasteful to use non-local
        points in the interpolator.

        We could select points to use in interpolation by using a
        boolean array computed from volumes == label, however it is
        much faster to index a large array with a slice object
        (self.slices).

        N.B. the slices don't isolate the volumes - each slice is
        only guaranteed to contain at least one complete volume and
        could capture parts of others. This means that we will have
        some overwriting of points where volumes overlap, possibly
        with nan.

        To overcome this problem we just re-run the interpolation on
        any remaining nans.
        """
        pp = self.run

        self.coords = [getattr(pp, c)[self.sub] * s for c, s in self.scales]
        self.data = [getattr(pp, d)[self.sub] for d in self.data_names]

        # TODO: determine invalid from full data
        # I think they are all coincident but we could use OR.
        # or we could check for coincidence and log if not.
        self.invalid = np.isnan(self.data[0])

        connections = np.ones((3, 3, 3))  # connect diagonals
        invalid_with_shell = ndi.binary_dilation(self.invalid,
                                                 structure=connections)
        self.complete_valid_shell = invalid_with_shell & ~self.invalid

        # label distinct invalid regions
        volumes, self.n = ndi.label(invalid_with_shell)
        # and find the slice that captures them
        self.slices = ndi.find_objects(volumes)
        # N.B. doesn't isolate volumes - see docstring

        # we can run into problems if the corners of the data array
        # are nan to start with, so we set them to their nearest
        # neighbour. We do this using a slice to select a corner
        # section of the array inside which to search.

        # slices that capture the outermost faces of the data array
        # on the vertical axis.
        # self.outer_slices = [slice(None, None, s - 1)
                             # for s in self.invalid.shape[:1]]

        # hopefully these are big enough to not capture all nans
        # if not, then we have bigger problems.
        cz, cx, ct = 2, 2, 5

        # slices that capture the eight corners of the 3d array
        self.outer_slices = [np.s_[:cz, :cx, :ct],
                             np.s_[:cz, :cx, -ct:],

                             np.s_[:cz, -cx:, :ct],
                             np.s_[:cz, -cx, -ct:],

                             np.s_[-cz:, :cx, :ct],
                             np.s_[-cz:, :cx, -ct:],

                             np.s_[-cz:, -cx:, :ct],
                             np.s_[-cz:, -cx, -ct:]]

    def construct_points(self, slice):
        """Find the valid shell within a given slice and return the
        coordinates and values of the points in the shell.
        """
        shell = self.complete_valid_shell[slice]

        valid_points = np.vstack(c[slice][shell] for c in self.coords).T
        valid_values = np.vstack(d[slice][shell] for d in self.data).T

        return valid_points, valid_values

    def compute_values(self, slice, nans, interpolator):
        """Compute the interpolated values for a given interpolator
        and the slice that it is valid inside.

        The interpolator isn't valid across the whole slice, just a
        region of nans contained within it, so we only evaluate the
        interpolator at the coordinates of those nans.
        """
        invalid_points = np.vstack(c[slice][nans] for c in self.coords).T
        invalid_values = interpolator(invalid_points).astype(np.float32)
        return invalid_values

    def evaluate_and_write(self, slice, interpolator):
        """For a given slice, compute the interpolated values and
        write them to the data array.

        Only writes where values are invalid to start with.
        """
        nans = self.invalid[slice]
        interpolated_values = self.compute_values(slice, nans, interpolator)

        for i, d in enumerate(self.data):
            # only copy across values that aren't nan
            values = interpolated_values[:, i]
            good = ~np.isnan(values)
            # see https://stackoverflow.com/questions/7179532
            d[slice].flat[np.flatnonzero(nans)[good]] = values[good]

    @property
    def nan_remaining(self):
        """How many nans are remaining in the data."""
        return sum(np.isnan(d).sum() for d in self.data)

    def process_parallel(self, processors=20):
        """Fill in the invalid regions of the data, using parallel
        processing to construct the interpolator for each distinct
        invalid region.

        The problem with single processing is that slow to calculate
        volumes block the execution of fast ones. It makes a lot of
        sense to multiprocess here.
        """
        self.process_outer()
        pool = mp.Pool(processes=processors)
        # arguments for interpolator construction
        valid_gen = ((slice,) + self.construct_points(slice)
                     for slice in self.slices)
        # need imap_unordered to avoid blocking. We have to pass any
        # state needed in post-processing though.
        interpolators = pool.imap_unordered(construct_interpolator, valid_gen)
        pool.close()

        for i, output in enumerate(interpolators):
            slice, interpolator = output
            print "Interpolating region # {: >5} / {}\r".format(i, self.n),
            sys.stdout.flush()
            self.evaluate_and_write(slice, interpolator)

        pool.join()

        print "\n"
        remaining = self.nan_remaining
        print "nans remaining: ", remaining
        sys.stdout.flush()

        # keep going until there are no more nans, which we should
        # have achieved on the first iteration.
        # TODO: stop if not converge?
        # when does it not converge?
        # If we only copy across non nan values we remove a step of
        # iteration but we still get caught by the literal edge
        # case that is causing the problem.
        # the edge case is that the index
        # (array([0, 0]), array([0, 1]), array([0, 0]))
        # will not get interpolated out.

        # this makes sense as there are no values on the outside of
        # the array to use for the interpolation.

        # ideas:
        # do nearest neighbour on the entire outside of the array

        if remaining != 0:
            self.setup()
            self.process_parallel()

    def process_outer(self):
        """Apply nearest neighbour interpolation to the corners of
        the array. You might wonder why we don't just use this
        method with a linear interpolator to do the whole thing: it
        is because we need to serialise for multi processing that we
        end up with a big class.
        """
        for slice in self.outer_slices:
            nans = self.invalid[slice]

            valid_points = np.vstack(c[slice][~nans] for c in self.coords).T
            valid_values = np.vstack(d[slice][~nans] for d in self.data).T

            interpolator = interp.NearestNDInterpolator(valid_points,
                                                        valid_values)

            invalid_points = np.vstack(c[slice][nans] for c in self.coords).T
            invalid_values = interpolator(invalid_points).astype(np.float32)

            for i, d in enumerate(self.data):
                # only copy across values that aren't nan
                values = invalid_values[:, i]
                good = ~np.isnan(values)
                # see https://stackoverflow.com/questions/7179532
                d[slice].flat[np.flatnonzero(nans)[good]] = values[good]

    def paint(self, processors=20):
        """Fill in the invalid (nan) regions of the data. """
        self.process_parallel(processors=processors)


def construct_interpolator((slice, coordinates, values)):
    """Construct a linear interpolator for given coordinates and values.

    This function is here because it needs to be pickleable for
    multiprocessing.

    slice is just passed through as state that we need for post
    processing the multiprocessing output.
    """
    return slice, interp.LinearNDInterpolator(coordinates, values)


def construct_nearest_interpolator((slice, coordinates, values)):
    return slice, interp.NearestNDInterpolator(coordinates, values)


def test():
    run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
    run.load()
    pp = g.PreProcessor(run)
    pp.extract_valid_region()
    pp.filter_zeroes()
    painter = Inpainter(pp, sub_region=np.s_[10:-10, :, :1000])
    painter.paint()


if __name__ == "__main__":
    test()
