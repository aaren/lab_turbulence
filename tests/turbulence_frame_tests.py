from nose.tools import *

import numpy as np

from gc_turbulence.turbulence import SingleLayerFrame, SingleLayerRun


columns_2d = np.dtype(SingleLayerRun.columns_2d)
columns_3d = np.dtype(SingleLayerRun.columns_3d)

csv_file = 'tests/ex_data/data/img.3b4olxqo.000500.csv'
txt_file = 'tests/ex_data/data/Export.3atnh4dp.000500.txt'
stereo_file = 'tests/ex_data/data/stereo_test.3eodh6wx.000011.txt'

csv_frame = SingleLayerFrame(fname=csv_file, columns=columns_2d)
txt_frame = SingleLayerFrame(fname=txt_file, columns=columns_2d)
stereo_frame = SingleLayerFrame(fname=stereo_file, columns=columns_3d)


def test_find_line_csv():
    """Check that we can determine where the header and content
    start.
    """
    assert_equal(csv_frame.header_line, 0)
    assert_equal(csv_frame.content_line, 7)


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


## Load the data to attributes of the frame
csv_frame.init_data()
txt_frame.init_data()
stereo_frame.init_data()


def test_data_read_csv():
    x, z, u, w = csv_frame.ix, csv_frame.iz, csv_frame.u_pix, csv_frame.w_pix
    assert_equal(x[0, 0], 0)
    assert_equal(z[0, 0], 0)
    assert_almost_equal(u[0, 0], -0.137607544660568)
    assert_almost_equal(w[0, 0], 0)


def test_data_read_txt():
    x, z, u, w = txt_frame.ix, txt_frame.iz, txt_frame.u_pix, txt_frame.w_pix
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


def test_timestamp_csv():
    """Check we are reading the timestamp correctly. Each frame has
    an array of the same dimension as the spatial arrays, with the
    timestamp in each element.
    """
    # same dimension as space arrays
    assert_tuple_equal(csv_frame.t.shape, csv_frame.x.shape)
    # timestamp in all elements
    assert((csv_frame.t == 5.00).all())
