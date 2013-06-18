from nose.tools import *

from ..gc_turbulence.turbulence import SingleLayer2dFrame
from ..gc_turbulence.turbulence import SingleLayer3dFrame


csv_file = 'tests/ex_data/data/img.3b4olxqo.000500.csv'
txt_file = 'tests/ex_data/data/Export.3atnh4dp.000500.txt'
stereo_file = 'tests/ex_data/data/stereo_test.3eodh6wx.000011.txt'

csv_frame = SingleLayer2dFrame(csv_file)
txt_frame = SingleLayer2dFrame(txt_file)
stereo_frame = SingleLayer3dFrame(stereo_file)


def test_header_csv():
    """Check that the csv file has header read
    properly.
    """
    assert_equal(csv_frame.header['FileID'], 'DSExport.CSV')


def test_header_txt():
    """Check that the text file has header read
    properly.
    """
    assert_equal(txt_frame.header['FileID'], 'DSExport.TAB')


def test_shape_csv():
    assert_equal(csv_frame.shape, (68, 95))


def test_shape_txt():
    assert_equal(txt_frame.shape, (68, 95))


def test_data_read_csv():
    x, z, u, w = csv_frame.x, csv_frame.z, csv_frame.u, csv_frame.w
    assert_equal(x[0, 0], 0)
    assert_equal(z[0, 0], 0)
    assert_almost_equal(u[0, 0], -0.137607544660568)
    assert_almost_equal(w[0, 0], 0)


def test_data_read_txt():
    x, z, u, w = txt_frame.x, txt_frame.z, txt_frame.u, txt_frame.w
    assert_equal(x[0, 0], 0)
    assert_equal(z[0, 0], 0)
    assert_almost_equal(u[0, 0], 0.400000005960464)
    assert_almost_equal(w[0, 0], 0)

def test_data_read_stereo():
    x = stereo_frame.x
    z = stereo_frame.z
    u = stereo_frame.u
    v = stereo_frame.v
    w = stereo_frame.w

    assert_equal(x[50, 50], 20.2136267196845)
    assert_equal(z[50, 50], 39.0232063188143)

    assert_almost_equal(u[50, 50], 0.00388681469485164)
    assert_almost_equal(w[50, 50], -0.00251218792982399)
    assert_almost_equal(v[50, 50], 0.00109881022945046)
