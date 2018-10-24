from processing import find_min_max
from processing import find_duration
from processing import find_peaks
from processing import find_num_beats
from processing import find_time_beats
from processing import find_bpm
from processing import createdictionary
import numpy as np
import pytest

def test_minmax():
    data=([1,0], [8,9], [-9999, 77], [0, 8], [0,77], [999, -55])
    answer = find_min_max(data)
    expected = (-55, 77)
    assert answer == expected

def test_minmax2():
    data= ([0,-1], [-1,0])
    answer = find_min_max(data)
    expected = (-1, 0)
    assert answer == expected

def test_findduration():
    data = ([0,1], [78,0], [1200, -5])
    answer = find_duration(data)
    expected = 1200
    assert answer == expected

def test_findpeaks():
    testdata = np.loadtxt("datatestprocess.csv", delimiter= ',')
    answer = find_peaks(testdata)
    np.testing.assert_array_equal(answer, ([19, 94, 170, 243, 306, 369]))

def test_countnumbeats():
    testdata= ([19, 94, 170, 243, 306, 369])
    answer = find_num_beats(testdata)
    expected = 6
    assert expected == answer

def test_countnumbeats2():
    with pytest.raises(TypeError):
        testdata = ()
        find_num_beats(testdata)


def test_findtimepeaks():
    data = np.loadtxt("datatestprocess.csv", delimiter= ',')
    peakindex= ([19, 94, 170, 243, 306, 369])
    peaktimes = find_time_beats(data,peakindex)
    np.testing.assert_array_almost_equal(peaktimes, ([1.583333, 7.833333, 14.166667, 20.416667, 26.716667, 33.016667]))

def test_findbpm():
    duration = 6000
    numbeats = 100
    bpm = find_bpm(duration, numbeats)
    assert bpm == 1

def test_dictionary():
    a= 5
    b= ([4, 3, 2])
    c= 3
    d= 'k'
    e= 1

    dict = createdictionary(a, b, c, d, e)

    assert dict == {'mean_hr_bpm' : 5,
                    'voltage_extremes': ([4, 3, 2]),
                    'duration': 3,
                    'num_beats': 'k',
                    'beats': 1}


