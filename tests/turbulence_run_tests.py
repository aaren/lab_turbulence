import os
from nose.tools import *

import numpy.testing as npt
from scipy.misc import imread

from ..gc_turbulence.turbulence import SingleLayer2dRun
from ..gc_turbulence.turbulence import SingleLayer2dFrame


baseline_quiver = 'tests/ex_data/baseline/quiver/quiver_000500.png'
w_dir = 'tests/ex_data'
rex = "000*"
ffmt = 'img.3b4olxqo.{rex}csv'.format(rex=rex)
run = SingleLayer2dRun(data_dir=w_dir, ffmt=ffmt, parallel=True)


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
    frame = SingleLayer2dFrame(run.files[0])
    npt.assert_array_equal(U[:, :, 0], frame.u)
