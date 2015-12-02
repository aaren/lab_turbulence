import sys
import logging
import multiprocessing as mp

import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp

import gc_turbulence as g


class ProcessOuterError(Exception):
    pass


class Inpainter(object):
    """Fills holes in gridded data (represented by nan) by
    performing linear interpolation using only the set of points
    surrounding each hole.
    """
    # maximum size of slice in time axis
    max_slice_size = 20
    # overlap between sub slices when a big slice is chopped up
    slice_overlap = 3

    def __init__(self, data, coords):
        """run has attributes representing coordinates (X, Z, T)
        and data (U, V, W).

        sub_region lets you optionally consider only a sub section
        of the data by indexing. Default is to take all of the data.

        scale is the factor to multiply the time coordinates by to
        make them comparable with the spatial coordinates in
        euclidean space. This should be a characteristic speed of
        the flow (e.g. the front velocity)
        """
        self.data = data
        self.coords = coords
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
        # TODO: determine invalid from full data
        # I think they are all coincident but we could use OR.
        # or we could check for coincidence and log if not.
        self.invalid = np.isnan(self.data[0])

        connections = np.ones((3, 3, 3))  # connect diagonals
        self.invalid_with_shell = ndi.binary_dilation(self.invalid,
                                                      structure=connections)
        self.complete_valid_shell = self.invalid_with_shell & ~self.invalid

        # label distinct invalid regions
        self.volumes, n = ndi.label(self.invalid_with_shell)
        # and find the slice that captures them
        slices = ndi.find_objects(self.volumes)
        # N.B. doesn't isolate volumes - see docstring

        self.slices = slices

        # big slices cause us problems so split them up into smaller
        # slices of a maximum width in time index
        # this actually hapens again in self.valid_points_generator, we
        # just do it here to find out the total length
        self.burst_slices = [s for original in slices
                             for s in self.burst(original)]
        self.n = len(self.burst_slices)

        # we can run into problems if the exterior of the data array
        # has nan to start with, so we set them to their nearest
        # neighbour. We do this using a slice to select the exterior
        # faces of the array.
        self.outer_slices = [slice(None, None, s - 1)
                             for s in self.invalid.shape]

        self.outer_slices = [np.s_[:2, :, 0:8000], np.s_[-2:, :, 0:8000],
                             np.s_[:, :2, 0:8000], np.s_[:, -2:, 0:8000],
                             np.s_[:, :, :2], np.s_[:, :, -2:],
                             ]

    def construct_points(self, slice, which):
        """Find the valid shell within a given slice and return the
        coordinates and values of the points in the shell.

        isvolume lets you define a mask that is used to further
        select points inside the slice.
        """
        valid_points = np.vstack(c[slice][which] for c in self.coords).T
        valid_values = np.vstack(d[slice][which] for d in self.data).T

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

    def burst(self, s, overlap=None, maxsize=None):
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
        overlap = overlap or self.slice_overlap
        maxsize = maxsize or self.max_slice_size

        sz, sx, st = s
        # define the left and right edges of the bins
        r = range(st.start, st.stop, maxsize)[1:] + [st.stop]
        l = [st.start] + [e - overlap for e in r[:-1]]
        # edge case comes when the list
        return [(sz, sx, slice(i, j, None)) for i, j in zip(l, r)]

    @property
    def valid_points_generator(self):
        """Construct valid points in a slice from the volume that
        created the slice.

        If the slice is large in extent in the time axis, it is
        burst into smaller slices.
        """
        for i, s in enumerate(self.slices):
            for slice in self.burst(s):
                # mask that selects the points in the burst slice
                # that are in the volume that the origin slice was
                # constructed to surround
                volume = self.volumes[slice] == i + 1
                # what if volume is empty (all false)?

                # mask that selects points that are within the slice AND
                # part of the shell of valid points AND part of the volume
                # that the slice is built around
                shell = self.complete_valid_shell[slice] & volume

                valid_points, valid_values = self.construct_points(slice,
                                                                   which=shell)
                yield (slice, valid_points, valid_values)

    def valid_outer_points_generator(self, slice_length):
        """Construct valid points in a slice from the volume that
        created the slice.

        If the slice is large in extent in the time axis, it is
        burst into smaller slices.
        """
        burst_slices = [self.burst(slice, maxsize=slice_length)
                        for slice in self.outer_slices[:-2]]
        burst_slices = np.array(burst_slices).reshape((-1, 3))
        burst_slices = np.vstack((burst_slices, self.outer_slices[-2:]))

        for slice in burst_slices:
            slice = list(slice)
            nans = self.invalid[slice]
            valid_points, valid_values = self.construct_points(slice,
                                                               which=~nans)
            yield (slice, valid_points, valid_values)

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
        print "inpainting by parallel method..."
        sys.stdout.flush()

        print "processing outer region..."
        sys.stdout.flush()
        self.process_outer(slice_lengths=[100, 200, 500,
                                          1000, 2000, 4000, 8000])
        print "done"
        sys.stdout.flush()

        # input and output queues
        input_stack = mp.Queue()
        output_stack = mp.Queue()

        pkwargs = {'target': self.parallel_processor,
                   'args':  (construct_linear_interpolator,
                             input_stack,
                             output_stack)}

        processes = [mp.Process(**pkwargs) for _ in range(processors)]

        print "Starting processes..."
        sys.stdout.flush()
        for p in processes:
            p.start()
        print len(processes), "Processes started and waiting"
        sys.stdout.flush()

        # populate the queue
        for data in self.valid_points_generator:
            input_stack.put(data)

        # add sentinels to stop processors
        for p in processes:
            input_stack.put('DONE')

        # read off the output stack as interpolators are calculated
        # and write to the data
        for i in range(self.n):
            print "Writing region # {: >5} / {}\r".format(i, self.n),
            sys.stdout.flush()
            slice, interpolator = output_stack.get()
            self.evaluate_and_write(slice, interpolator)

        # wait to finish
        for p in processes:
            p.join()

        print "\nnans remaining: ", self.nan_remaining
        sys.stdout.flush()

        # keep going until no more nans or we reach the recursion limit
        if self.nan_remaining != 0 and recursion:
            self.setup()
            self.process_parallel(processors, recursion - 1)

    @staticmethod
    def parallel_processor(function, inputs, outputs, sentinel='DONE'):
        """Simple threading processor that reads from the inputs
        queue and calls function on what it gets. The output
        from function(input) is written to the outputs queue.

        Stops when it encounters a sentinel on the input queue.
        """
        while True:
            data = inputs.get()
            if data == sentinel:
                break
            else:
                outputs.put(function(data))

    def process_serial(self):
        """Single core interpolation"""
        print "\n"
        self.process_outer()

        for args in self.valid_points_generator:
            print "\rInterpolation over {}".format(args[0]),
            sys.stdout.flush()
            slice, interpolator = construct_linear_interpolator(args)
            self.evaluate_and_write(slice, interpolator)

        print "\n"
        remaining = self.nan_remaining
        print "nans remaining: ", remaining
        sys.stdout.flush()

    def process_outer(self, slice_lengths):
        """Apply nearest neighbour interpolation to the corners of
        the array. You might wonder why we don't just use this
        method with a linear interpolator to do the whole thing: it
        is because we need to serialise for multi processing that we
        end up with a big class.
        """
        try:
            error = False
            length = slice_lengths[0]
            logging.info('Attempting process_outer with '
                         'slice_length={}'.format(length))
            for args in self.valid_outer_points_generator(slice_length=length):
                print "\rInterpolation over {}".format(args[0]),
                sys.stdout.flush()
                try:
                    slice, interpolator = construct_nearest_interpolator(args)
                    self.evaluate_and_write(slice, interpolator)
                except ValueError:
                    error = True
                    continue
        except IndexError:
            # ran out of slice lengths
            logging.exception('Could not create valid outer shell. '
                              'Tried {}'.format(slice_lengths))
            raise ProcessOuterError

        if error is True:
            # no points for the interpolator somewhere
            slice_lengths.pop(0)
            self.process_outer(slice_lengths)

    def process_outer_old(self):
        """Apply nearest neighbour interpolation to the corners of
        the array. You might wonder why we don't just use this
        method with a linear interpolator to do the whole thing: it
        is because we need to serialise for multi processing that we
        end up with a big class.
        """
        for slice in self.outer_slices:
            logging.info('using {}'.format(slice))
            nans = self.invalid[slice]
            logging.info('found {} nans'.format(nans.sum()))

            valid_points = np.vstack(c[slice][~nans] for c in self.coords).T
            valid_values = np.vstack(d[slice][~nans] for d in self.data).T

            dim = valid_points.shape
            logging.info('creating interpolator over {}'.format(dim))

            interpolator = interp.NearestNDInterpolator(valid_points,
                                                        valid_values)

            invalid_points = np.vstack(c[slice][nans] for c in self.coords).T
            idim = invalid_points.shape
            logging.info('interpolating over {}'.format(idim))
            invalid_values = interpolator(invalid_points).astype(np.float32)

            logging.info('copying data')
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


def construct_nearest_interpolator((slice, coordinates, values)):
    """Construct a linear interpolator for given coordinates and values.

    This function is here because it needs to be pickleable for
    multiprocessing.

    slice is just passed through as state that we need for post
    processing the multiprocessing output.
    """
    return slice, interp.NearestNDInterpolator(coordinates, values)


def construct_linear_interpolator((slice, coordinates, values)):
    """Construct a linear interpolator for given coordinates and values.

    This function is here because it needs to be pickleable for
    multiprocessing.

    slice is just passed through as state that we need for post
    processing the multiprocessing output.
    """
    return slice, interp.LinearNDInterpolator(coordinates, values)


def test():
    run = g.RawRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
    run.load()
    pp = g.PreProcessor(run)
    pp.extract_valid_region()
    pp.filter_zeroes()
    painter = Inpainter(pp, sub_region=np.s_[10:-10, :, :1000])
    painter.paint()


if __name__ == "__main__":
    test()
