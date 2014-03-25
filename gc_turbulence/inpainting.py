import sys
import multiprocessing as mp

import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp

import gc_turbulence as g


class Inpainter(object):
    def __init__(self, run):
        self.run = run
        self.setup()

    def setup(self):
        # pp.interpolate_zeroes()
        pp = self.run
        self.coords = pp.Z, pp.X, pp.T
        # TODO: use all data
        self.data = pp.U

        # TODO: determine invalid from full data
        # I think they are all coincident but we could use OR.
        # or we could check for coincidence and log if not.
        self.invalid = np.isnan(pp.U)
        # connect diagonals
        connections = np.ones((3, 3, 3))
        invalid_with_shell = ndi.binary_dilation(self.invalid,
                                                 structure=connections)
        self.complete_valid_shell = invalid_with_shell & ~self.invalid

        # label distinct invalid regions
        volumes, self.n = ndi.label(invalid_with_shell)
        # and find the slice that captures them
        self.slices = ndi.find_objects(volumes)
        # N.B this doesn't isolate the volumes - each slice is only
        # guaranteed to contain at least one complete volume and
        # could capture parts of others.

    def process_parallel(self):
        print self.n
        pool = mp.Pool(processes=20)
        valid_gen = (self.construct_valid(slice) for slice in self.slices)
        interpolators = pool.imap_unordered(construct_interpolator, valid_gen)
        pool.close()

        # TODO: try single processing - is it worth the overhead?
        # The problem with single processing is that slow to calculate
        # volumes block the execution of fast ones. It makes a lot of
        # sense to multiprocess here.

        for i, output in enumerate(interpolators):
            slice, interpolator = output
            print "# {} {}\r".format(i, slice),
            sys.stdout.flush()

            nans = self.invalid[slice]
            invalid_points = np.vstack(c[slice][nans] for c in self.coords).T
            invalid_values = interpolator(invalid_points).astype(np.float32)

            # TODO: assign v, w
            self.data[slice][nans] = invalid_values

        pool.join()

        print "\n"
        nan_remaining = np.where(np.isnan(self.data))[0].size
        print "nans remaining: ", nan_remaining

        # keep going until there are no more nans
        if nan_remaining != 0:
            self.setup()
            self.process_parallel()

    def construct_valid(self, slice):
        shell = self.complete_valid_shell[slice]

        valid_points = np.vstack(c[slice][shell] for c in self.coords).T
        valid_values = self.data[slice][shell]

        return slice, valid_points, valid_values


def construct_interpolator((slice, points, values)):
    return slice, interp.LinearNDInterpolator(points, values)


def test():
    run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
    run.load()
    pp = g.PreProcessor(run)
    pp.extract_valid_region()
    pp.filter_zeroes()
    painter = Inpainter(pp)
    painter.process_parallel()


if __name__ == "__main__":
    test()
