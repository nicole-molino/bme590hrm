from reader import readinfile
import pytest

def test_reader():
    (time, voltage) = readinfile('readertestdata.csv')
    assert time == [1.0, 9.0, 4.4, 1.44]
    assert voltage == [12.0, 18.0, 3.0, 4.0]
    


