from nose.tools import *

# from gc_turbulence import turbulence
# import SingleLayer2dFrame
# from gc_turbulence.turbulence import SingleLayer2dFrame
# import gc_turbulence
from gc_turbulence import SingleLayer2dFrame


csv_file = 'tests/ex_data/data/img.3b4olxqo.000500.csv'
txt_file = 'tests/ex_data/data/Export.3atnh4dp.000500.txt'

csv_frame = SingleLayer2dFrame(csv_file)
txt_frame = SingleLayer2dFrame(txt_file)


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
