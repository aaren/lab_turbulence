import os
from nose.tools import *

import numpy.testing as npt
import numpy
import h5py

from ..gc_turbulence.turbulence import SingleLayerFrame
from ..gc_turbulence.turbulence import SingleLayerRun


# delete all cache
# for root, dirs, files in os.walk('tests/ex_data/cache', topdown=False):
    # for name in files:
        # os.remove(os.path.join(root, name))
    # for name in dirs:
        # os.rmdir(os.path.join(root, name))

data_dir = 'tests/ex_data/data'
cache_dir = 'tests/ex_data/cache'
valid_cache = 'tests/ex_data/cache/valid_stereo_cache.hdf5'
cache_test = 'tests/ex_data/cache/test_cache.hdf5'

if os.path.exists(cache_test):
    os.remove(cache_test)

run_kwargs = dict(data_dir=data_dir, pattern='3b4olxqo', rex='.000*',
                  stereo=False)
run = SingleLayerRun(**run_kwargs)
stereo_run_kwargs = dict(data_dir=data_dir, pattern='3eodh6wx', rex='*',
                         stereo=True)
stereo_run = SingleLayerRun(**stereo_run_kwargs)

columns_2d = SingleLayerRun.columns_2d
columns_3d = SingleLayerRun.columns_3d


## All runs need to be imported to hdf5 before they can be accessed
## There is a test hdf5 in valid_cache. Let's try loading this.
def test_load():
    r = SingleLayerRun(cache_path=valid_cache, stereo=True)
    r.load()
    assert(type(r.u) is h5py._hl.dataset.Dataset)
    assert_equal(r.u.shape, (68, 86, 10))
    r.load_to_memory()
    assert(type(r.u) is numpy.ndarray)


def test_import_run():
    """Can we import from text files?"""
    stereo_run.cache_path = os.path.join(cache_test)
    stereo_run.import_to_hdf5()
    print stereo_run.valid_cache_exists
    print stereo_run.cache_path
    stereo_run.load()
    valid_run = SingleLayerRun(cache_path=valid_cache)

    npt.assert_array_equal(stereo_run.u[...], valid_run.u[...])


# def test_frames():
    # """Generates an array of horizontal velocities from the test data."""
    # U = run.u
    # assert_equal(U.shape[-1], run.nfiles)
    # frame = SingleLayerFrame(fname=run.files[0], columns=columns_2d)
    # npt.assert_array_equal(U[:, :, 0], frame.u)


# def test_stereo_frames():
    # """Test the reading of stereo data."""
    # for vel in ('u', 'v', 'w'):
        # U = getattr(stereo_run, vel)
        # assert_equal(U.shape[-1], stereo_run.nfiles)
        # frame = SingleLayerFrame(fname=stereo_run.files[0], columns=columns_3d)
        # assert_equal(frame.fname, stereo_run.files[0])
        # u = getattr(frame, vel)
        # npt.assert_array_equal(U[:, :, 0], u)


# def test_timestamps():
    # """Depth of time array should be the same as number of files."""
    # assert_equal(run.nfiles, run.t.shape[2])
