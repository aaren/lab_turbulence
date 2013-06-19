import os
from nose.tools import *

import numpy.testing as npt
from scipy.misc import imread

from ..gc_turbulence.turbulence import SingleLayerFrame
from ..gc_turbulence.turbulence import SingleLayer2dRun
from ..gc_turbulence.turbulence import SingleLayer3dRun


baseline_quiver = 'tests/ex_data/baseline/quiver/quiver_000500.png'
w_dir = 'tests/ex_data'

run_kwargs = dict(data_dir=w_dir, index='3b4olxqo', rex='.000*')
run = SingleLayer2dRun(**run_kwargs)
stereo_run_kwargs = dict(data_dir=w_dir, index='3eodh6wx', rex='.00001*')
stereo_run = SingleLayer3dRun(**stereo_run_kwargs)


def test_quivers():
    """Make a load of quiver plots with multiprocessing (default)
    and compare one of the output images to a reference image.
    """
    run.make_quivers()
    test_quiver = os.path.join(w_dir, 'quiver/quiver_000500.png')
    test_im = imread(test_quiver)
    baseline_im = imread(baseline_quiver)
    npt.assert_array_equal(test_im, baseline_im)


def test_frames():
    """Generates an array of horizontal velocities from the test data."""
    U = run.U
    assert_equal(U.shape[-1], run.nfiles)
    frame = SingleLayerFrame(fname=run.files[0])
    npt.assert_array_equal(U[:, :, 0], frame.u)


def test_stereo_frames():
    """Test the reading of stereo data."""
    for vel in ('U', 'V', 'W'):
        U = getattr(stereo_run, vel)
        assert_equal(U.shape[-1], stereo_run.nfiles)
        frame = SingleLayerFrame(fname=stereo_run.files[0], stereo=True)
        assert_equal(frame.fname, stereo_run.files[0])
        u = getattr(frame, vel.lower())
        npt.assert_array_equal(U[:, :, 0], u)


def test_pickle_run():
    """Write a run object to disk and load it up again."""
    # instantiate run, forcing reload (i.e. not use pickled data)
    run = SingleLayer2dRun(cache_dir='tests/ex_data/cache',
                           caching=False,
                           **run_kwargs)
    # save to pickle file
    run.save()
    # U should not be loaded when we load from the pickle file
    run2 = SingleLayer2dRun(cache_dir='tests/ex_data/cache',
                            caching=True,
                            **run_kwargs)
    frames_attr_name = '_lazy_frames'
    assert(not(hasattr(run2, frames_attr_name)))

    # load U and save to pickle file
    U = run.U
    run.save()
    # U should now be loaded
    run2 = SingleLayer2dRun(cache_dir='tests/ex_data/cache',
                            caching=True,
                            **run_kwargs)
    assert(hasattr(run2, frames_attr_name))

    # compare the run and the saved copy
    U2 = run2.U
    npt.assert_array_equal(U, U2)

    # delete the cache file
    os.remove(run2.cache_path)
    # U should not be loaded
    run3 = SingleLayer2dRun(cache_dir='tests/ex_data/cache',
                            caching=True,
                            **run_kwargs)
    assert(not(hasattr(run3, frames_attr_name)))


def test_pickle_run_autosave():
    """When frames are loaded, run should save automatically."""

    frames_attr_name = '_lazy_frames'

    # create a fresh run, no frames loaded (default)
    run = SingleLayer2dRun(cache_dir='tests/ex_data/cache',
                           caching=False,
                           **run_kwargs)
    # overwrite cache file with no frames
    run.save()
    # reload False and cache file will load at instantiate
    cache_run = SingleLayer2dRun(cache_dir='tests/ex_data/cache',
                                 caching=True,
                                 **run_kwargs)
    # check no frames in both runs
    assert(not(hasattr(run, frames_attr_name)))
    assert(not(hasattr(cache_run, frames_attr_name)))

    # this should not overwrite cache file with frames as caching
    # disabled
    run.U
    cache_run.load()
    assert(not(hasattr(cache_run, frames_attr_name)))
    # forcing the save will write out to cache file
    run.save()
    cache_run.load()
    assert(hasattr(cache_run, frames_attr_name))

    # turn on caching
    run.toggle_cache(True)
    assert(run.caching)
    run.U
    # run should now have lazy_frames
    assert(hasattr(run, frames_attr_name))
    # load the cache file and check for frames
    cache_run.load()
    assert(hasattr(cache_run, frames_attr_name))


def test_reload():
    run.reload()
