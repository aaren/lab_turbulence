from nose.tools import *

from gc_turbulence.turbulence import SingleLayer2dFrame
# from turbulence import SingleLayer2dRun


csv_file = 'tests/ex_data/img.3b4olxqo.000200.csv'
txt_file = 'tests/ex_data/Export.3atnh4dp.000500.txt'

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
    x, y, u, v = csv_frame.data
    assert_equal(x[0, 0], 0)
    assert_equal(y[0, 0], 0)
    assert_almost_equal(u[0, 0], 0.35160830616951)
    assert_almost_equal(v[0, 0], 0)


def test_data_read_txt():
    x, y, u, v = txt_frame.data
    assert_equal(x[0, 0], 0)
    assert_equal(y[0, 0], 0)
    assert_almost_equal(u[0, 0], 0.400000005960464)
    assert_almost_equal(v[0, 0], 0)
