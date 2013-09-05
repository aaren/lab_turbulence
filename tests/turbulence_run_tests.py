import os
import sys

from nose.tools import *

import numpy.testing as npt
from scipy.misc import imread

from gc_turbulence.turbulence import SingleLayerFrame
from gc_turbulence.turbulence import SingleLayerRun


# delete all cache
for root, dirs, files in os.walk('tests/ex_data/cache', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))

w_dir = 'tests/ex_data'

run_kwargs = dict(data_dir=w_dir, index='3b4olxqo', rex='.000*')
run = SingleLayerRun(**run_kwargs)
stereo_run_kwargs = dict(data_dir=w_dir, index='3eodh6wx', rex='.00001*', stereo=True)
stereo_run = SingleLayerRun(**stereo_run_kwargs)

columns_2d = {'x': 0,
              'z': 1,
              'u': 6,
              'w': 7}

columns_3d = {'x': 2,
              'z': 3,
              'u': 4,
              'v': 6,
              'w': 5}
sys.stderr.write('...init')

def test_frames():
    """Generates an array of horizontal velocities from the test data."""
    U = run.U
    assert_equal(U.shape[-1], run.nfiles)
    frame = SingleLayerFrame(fname=run.files[0], columns=columns_2d)
    npt.assert_array_equal(U[:, :, 0], frame.u)


def test_stereo_frames():
    """Test the reading of stereo data."""
    for vel in ('U', 'V', 'W'):
        U = getattr(stereo_run, vel)
        assert_equal(U.shape[-1], stereo_run.nfiles)
        frame = SingleLayerFrame(fname=stereo_run.files[0], columns=columns_3d)
        assert_equal(frame.fname, stereo_run.files[0])
        u = getattr(frame, vel.lower())
        npt.assert_array_equal(U[:, :, 0], u)


def test_save_run():
    """Write a run object to disk and load it up again."""
    # instantiate run, forcing reload (i.e. not use pickled data)
    run = SingleLayerRun(cache_dir='tests/ex_data/cache',
                         caching=False,
                         **run_kwargs)

    # save to npz file
    run.save()
    # U should be loaded when we load from the npz file
    run2 = SingleLayerRun(cache_dir='tests/ex_data/cache',
                          caching=True,
                          **run_kwargs)
    U_attr_name = 'U'
    assert(hasattr(run2, U_attr_name))

    # compare the run and the saved copy
    npt.assert_array_equal(run.U, run2.U)

    # delete the cache file
    os.remove(run2.cache_path)
    # should get UserWarning when try and load with caching enabled
    assert_raises(UserWarning,
                  SingleLayerRun,
                  cache_dir='tests/ex_data/cache',
                  caching=True,
                  **run_kwargs)


def test_pickle_run_autosave():
    """When frames are loaded, run should save automatically."""
    #TODO: write me!

def test_reload():
    run.reload()


def test_timestamps():
    """Depth of time array should be the same as number of files."""
    assert_equal(run.nfiles, run.T.shape[2])
