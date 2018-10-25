from reader import readinfile
import pytest
import numpy

def test_reader():
    data = readinfile('readertestdata.csv')
    numpy.testing.assert_array_equal(data, ([1.0, 12], [9, 18], [4.4, 3], [6, 4], [1.44, 6]))

def test_reader_ValueError():
    with pytest.raises(ValueError):
        data = readinfile('readertestdatastrings.csv')
