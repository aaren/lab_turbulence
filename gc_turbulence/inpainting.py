import sys
import multiprocessing as mp

import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp

import gc_turbulence as g


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
        self.volumes, n = ndi.label(invalid_with_shell)
        # and find the slice that captures them
        slices = ndi.find_objects(self.volumes)
        # N.B. doesn't isolate volumes - see docstring

        # big slices cause us problems so split them up into smaller
        # slices of a maximum width in time index
        def burst(s, dt=20, overlap=3):
            """Split a three axis slice into a number of smaller
            slices with a maximum size on the third axis and an
            overlap.

            The overlap is there for an edge case in Qhull that
            occurs when the slice is only 1 element long in an axis.

            This is a problem because then the points are coplanar
            and Qhull gets confused. This method guarantees that we
            get an overlap between slices and no slice thinner than
            the overlap unless it is the only slice.
            """
            sz, sx, st = s
            # define the left and right edges of the bins
            r = range(st.start, st.stop, dt)[1:] + [st.stop]
            l = [st.start] + [e - overlap for e in r[:-1]]
            # edge case comes when the list
            return [(sz, sx, slice(i, j, None)) for i, j in zip(l, r)]

        self.slices = [s for original in slices for s in burst(original)]
        self.n = len(self.slices)

        # we can run into problems if the exterior of the data array
        # has nan to start with, so we set them to their nearest
        # neighbour. We do this using a slice to select the exterior
        # faces of the array.
        self.outer_slices = [slice(None, None, s - 1)
                             for s in self.invalid.shape]

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

    def process_parallel(self, processors=20, recursion=0):
        """Fill in the invalid regions of the data, using parallel
        processing to construct the interpolator for each distinct
        invalid region.

        The problem with single processing is that slow to calculate
        volumes block the execution of fast ones. It makes a lot of
        sense to multiprocess here.

        recursion (integer) is the number of recursions that are
        allowed to happen. Setting 0 or False will have no
        recursion. Setting it to -1 will recurse forever.
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
        if remaining != 0 and recursion:
            self.setup()
            self.process_parallel(processors, recursion - 1)

    def process_serial(self):
        """Single core interpolation"""
        print "\n"
        self.process_outer()
        for slice in self.slices:
            print "\rInterpolation over {}".format(slice),
            sys.stdout.flush()
            valid_points, valid_values = self.construct_points(slice)
            _, interpolator = construct_interpolator((slice,
                                                      valid_points,
                                                      valid_values))
            self.evaluate_and_write(slice, interpolator)

        print "\n"
        remaining = self.nan_remaining
        print "nans remaining: ", remaining
        sys.stdout.flush()

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

    def paint(self, processors=20, recursion=5):
        """Fill in the invalid (nan) regions of the data. """
        if processors == 1:
            self.process_serial()
        else:
            self.process_parallel(processors=processors, recursion=recursion)


def construct_interpolator((slice, coordinates, values)):
    """Construct a linear interpolator for given coordinates and values.

    This function is here because it needs to be pickleable for
    multiprocessing.

    slice is just passed through as state that we need for post
    processing the multiprocessing output.
    """
    return slice, interp.LinearNDInterpolator(coordinates, values)


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
