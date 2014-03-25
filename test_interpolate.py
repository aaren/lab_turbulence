import sys
import multiprocessing as mp

import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp

import gc_turbulence as g

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()
# pp.interpolate_zeroes()

coords = pp.Z, pp.X, pp.T
data = pp.U, pp.V, pp.W

all_coords = np.concatenate([c[None] for c in coords])

data = pp.U[:, :, 2000:-2000]

invalid = np.isnan(data)
invalid_with_shell = ndi.binary_dilation(invalid, iterations=1, structure=np.ones((3, 3, 3)))
complete_valid_shell = invalid_with_shell & ~invalid

volumes, n = ndi.label(invalid_with_shell)
slices = ndi.find_objects(volumes)



def interpolate_region(slice):
    nans = invalid[slice]
    shell = complete_valid_shell[slice]

    valid_points = np.vstack(c[slice][shell] for c in coords).T
    valid_values = data[slice][shell]

    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)

    invalid_points = np.vstack(c[slice][nans] for c in coords).T
    invalid_values = interpolator(invalid_points).astype(valid_values.dtype)

    return slice, invalid_values


def full_state_interpolate_region((slice, complete_valid_shell, invalid)):
    nans = invalid[slice]
    shell = complete_valid_shell[slice]

    valid_points = np.vstack(c[slice][shell] for c in coords).T
    valid_values = data[slice][shell]

    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)

    invalid_points = np.vstack(c[slice][nans] for c in coords).T
    invalid_values = interpolator(invalid_points).astype(valid_values.dtype)

    return slice, invalid_values


def fillin(volume):
    """Construct the interpolator for a single shell and evaluate
    inside."""
    # construct the interpolator for a single shell
    shell = volume & ~invalid
    inside_shell = volume & invalid

    valid_points = all_coords[..., shell].T
    # valid_values = np.vstack(d[shell] for d in data).T
    valid_values = data[shell]

    # coordinates of all of the invalid points
    invalid_points = all_coords[..., inside_shell].T

    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)
    # this is nan outside of the shell
    invalid_values = interpolator(invalid_points)
    return volume, invalid_values


def main_fillin():
    print n
    args = (volumes == i for i in xrange(1, n))

    pool = mp.Pool(processes=20)
    result = pool.imap_unordered(fillin, args)
    pool.close()
    for i, output in enumerate(result):
        print "# {}\r".format(i),
        sys.stdout.flush()
        volume, invalid_values = output
        inside_shell = volume & invalid
        data[inside_shell] = invalid_values
    pool.join()
    print "\n"
    print "nans remaining: ", np.where(np.isnan(data))[0].size


def construct_valid(slice):
    shell = complete_valid_shell[slice]

    valid_points = np.vstack(c[slice][shell] for c in coords).T
    valid_values = data[slice][shell]

    return slice, valid_points, valid_values


def construct_interpolator((slice, points, values)):
    return slice, interp.LinearNDInterpolator(points, values)


def full_construct_interpolator((complete_valid_shell, slice)):
    shell = complete_valid_shell[slice]

    valid_points = np.vstack(c[slice][shell] for c in coords).T
    valid_values = data[slice][shell]
    return slice, interp.LinearNDInterpolator(valid_points, valid_values)


def alt_main_parallel():
    global n
    print n

    pool = mp.Pool(processes=20)
    valid_gen = (construct_valid(slice) for slice in slices)
    interpolators = pool.imap_unordered(construct_interpolator, valid_gen)

    # can't use imap_unordered if we're relying on ordered output
    # later so we keep the slice with the interpolator
    # interpolators = pool.imap(construct_interpolator, valid_gen)

    pool.close()

    for i, output in enumerate(interpolators):
        slice, interpolator = output
        print "# {} {}\r".format(i, slice),
        sys.stdout.flush()

        nans = invalid[slice]
        invalid_points = np.vstack(c[slice][nans] for c in coords).T
        invalid_values = interpolator(invalid_points).astype(np.float32)

        data[slice][nans] = invalid_values

    pool.join()
    print "\n"
    nan_remaining = np.where(np.isnan(data))[0].size
    print "nans remaining: ", nan_remaining
    if nan_remaining != 0:
        global invalid
        global complete_valid_shell
        global slices
        invalid = np.isnan(data)
        invalid_with_shell = ndi.binary_dilation(invalid, iterations=1, structure=np.ones((3, 3, 3)))
        complete_valid_shell = invalid_with_shell & ~invalid

        labels, n = ndi.label(invalid_with_shell)
        slices = ndi.find_objects(labels)

        alt_main_parallel()


def alt_alt_main_parallel():
    global n
    print n

    pool = mp.Pool(processes=20)

    input_gen = ((complete_valid_shell, slice) for slice in slices)
    interpolators = pool.imap_unordered(full_construct_interpolator, input_gen)
    # can't use imap_unordered if we're relying on ordered output
    # later so we keep the slice with the interpolator
    # interpolators = pool.imap(construct_interpolator, valid_gen)

    pool.close()

    for i, output in enumerate(interpolators):
        slice, interpolator = output
        print "# {} {}\r".format(i, slice),
        sys.stdout.flush()

        nans = invalid[slice]
        invalid_points = np.vstack(c[slice][nans] for c in coords).T
        invalid_values = interpolator(invalid_points).astype(np.float32)

        data[slice][nans] = invalid_values

    pool.join()
    print "\n"
    nan_remaining = np.where(np.isnan(data))[0].size
    print "nans remaining: ", nan_remaining
    if nan_remaining != 0:
        global invalid
        global complete_valid_shell
        global slices
        invalid = np.isnan(data)
        invalid_with_shell = ndi.binary_dilation(invalid, iterations=1, structure=np.ones((3, 3, 3)))
        complete_valid_shell = invalid_with_shell & ~invalid

        labels, n = ndi.label(invalid_with_shell)
        slices = ndi.find_objects(labels)

        alt_main_parallel()


def main_parallel():
    global n
    print n
    pool = mp.Pool(processes=20)
    result = pool.imap_unordered(interpolate_region, slices)

    # TODO: try single processing - is it worth the overhead?
    # The problem with single processing is that slow to calculate
    # volumes block the execution of fast ones. It makes a lot of
    # sense to multiprocess here.

    pool.close()
    for i, output in enumerate(result):
        slice, invalid_values = output
        print "# {} {}\r".format(i, slice),
        sys.stdout.flush()
        nans = invalid[slice]
        data[slice][nans] = invalid_values

    pool.join()
    print "\n"
    nan_remaining = np.where(np.isnan(data))[0].size
    print "nans remaining: ", nan_remaining
    if nan_remaining != 0:
        global invalid
        global complete_valid_shell
        global slices
        invalid = np.isnan(data)
        invalid_with_shell = ndi.binary_dilation(invalid, iterations=1, structure=np.ones((3, 3, 3)))
        complete_valid_shell = invalid_with_shell & ~invalid

        labels, n = ndi.label(invalid_with_shell)
        slices = ndi.find_objects(labels)

        main_parallel()


def full_state_main_parallel():
    global n
    print n
    pool = mp.Pool(processes=20)
    args = ((slice, complete_valid_shell, invalid) for slice in slices)
    result = pool.imap_unordered(full_state_interpolate_region, args)

    # TODO: try single processing - is it worth the overhead?
    # The problem with single processing is that slow to calculate
    # volumes block the execution of fast ones. It makes a lot of
    # sense to multiprocess here.

    pool.close()
    for i, output in enumerate(result):
        slice, invalid_values = output
        print "# {} {}\r".format(i, slice),
        sys.stdout.flush()
        nans = invalid[slice]
        data[slice][nans] = invalid_values

    pool.join()
    print "\n"
    nan_remaining = np.where(np.isnan(data))[0].size
    print "nans remaining: ", nan_remaining
    if nan_remaining != 0:
        global invalid
        global complete_valid_shell
        global slices
        invalid = np.isnan(data)
        invalid_with_shell = ndi.binary_dilation(invalid, iterations=1, structure=np.ones((3, 3, 3)))
        complete_valid_shell = invalid_with_shell & ~invalid

        labels, n = ndi.label(invalid_with_shell)
        slices = ndi.find_objects(labels)

        main_parallel()


def main_single():
    global n
    print n
    import itertools
    result = itertools.imap(interpolate_region, slices)

    # TODO: try single processing - is it worth the overhead?

    for i, output in enumerate(result):
        slice, invalid_values = output
        print "# {} {}\r".format(i, slice),
        sys.stdout.flush()
        nans = invalid[slice]
        data[slice][nans] = invalid_values

    print "\n"
    nan_remaining = np.where(np.isnan(data))[0].size
    print "nans remaining: ", nan_remaining
    if nan_remaining != 0:
        global invalid
        global complete_valid_shell
        global slices
        invalid = np.isnan(data)
        invalid_with_shell = ndi.binary_dilation(invalid, iterations=1, structure=np.ones((3, 3, 3)))
        complete_valid_shell = invalid_with_shell & ~invalid

        labels, n = ndi.label(invalid_with_shell)
        slices = ndi.find_objects(labels)

        main_single()


def main():
    for i, slice in enumerate(slices):
        print "# {}\r".format(i),
        sys.stdout.flush()
        slice, invalid_values = interpolate_region(slice)
        nans = invalid[slice]
        data[slice][nans] = invalid_values


if __name__ == '__main__':
    func = sys.argv[1]
    locals()[func]()
    # 2m10 to complete laptop
    # main_parallel()

    # 2m58 to complete laptop
    # alt_main_parallel()

    # 3m31 to complete laptop
    # alt_alt_main_parallel()

    # 3m56 to complete laptop
    # full_state_main_parallel()

    # main_single()
    # main_fillin()
