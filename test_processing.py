from processing import find_min_max
from processing import find_duration
from processing import find_peaks
from processing import find_num_beats_per_window
from processing import find_total_numbeats
from processing import find_time_beats
from processing import find_bpm
from processing import createdictionary
from processing import find_windowsize
from processing import define_windows

import numpy as np
import pytest


def test_minmax():
    data = ([1, 0], [8, 9], [-9999, 77], [0, 8], [0, 77], [999, -55])
    answer = find_min_max(data)
    expected = [-55, 77]
    assert answer == expected


def test_minmax2():
    data = ([0, -1], [-1, 0])
    answer = find_min_max(data)
    expected = [-1, 0]
    assert answer == expected


def test_findduration():
    data = ([0, 1], [78, 0], [1200, -5])
    answer = find_duration(data)
    expected = 1200
    assert answer == expected


def test_splitwindows1():
    data = (
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
        [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24],
        [25, 26], [27, 28], [29, 30], [31, 32])
    w = find_windowsize(data)
    assert w == 3


def test_findwindows():
    data = np.array(
        [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
         (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
         (25, 26), (27, 28), (29, 30), (31, 32)])
    w = 3

    (data_w1, data_w2, data_w3, data_w4, data_w5,
     data_w6) = define_windows(w, data)

    np.testing.assert_array_equal(data_w1, np.array([(1, 2),
                                                     (3, 4), (5, 6)]))
    np.testing.assert_array_equal(data_w2, np.array([(7, 8),
                                                     (9, 10), (11, 12)]))
    np.testing.assert_array_equal(data_w3, np.array([(13, 14),
                                                     (15, 16), (17, 18)]))
    np.testing.assert_array_equal(data_w4, np.array([(19, 20),
                                                     (21, 22), (23, 24)]))
    np.testing.assert_array_equal(data_w5, np.array([(25, 26),
                                                     (27, 28), (29, 30)]))
    np.testing.assert_array_equal(data_w6, np.array([[31, 32]]))


def test_findpeaks():
    testdata = np.loadtxt("datatestprocess.csv", delimiter=',')

    w = 63
    dw1 = testdata[0:w, :]
    dw2 = testdata[w:w * 2, :]
    dw3 = testdata[w * 2:w * 3, :]
    dw4 = testdata[w * 3:w * 4, :]
    dw5 = testdata[w * 4:w * 5, :]
    dw6 = testdata[w * 5:, :]

    (p1, p2, p3, p4, p5, p6) = find_peaks(dw1, dw2, dw3, dw4, dw5, dw6)

    np.testing.assert_array_equal(p1, 19)
    np.testing.assert_array_equal(p2, 31)
    np.testing.assert_array_equal(p3, 44)
    np.testing.assert_array_equal(p4, 54)
    np.testing.assert_array_equal(p5, 54)
    np.testing.assert_array_equal(p6, 54)


def test_countnumbeats():
    testdata = np.loadtxt("datatestprocess.csv", delimiter=',')

    p1 = [1, 2]
    p2 = [1, 3, 4, 4]
    p3 = [1, 2, 3, 5, 8]
    p4 = [1]
    p5 = [1, 2, 3, 5, 8]
    p6 = [1, 2, 3, 5, 8, 8]

    (a1, a2, a3, a4, a5, a6) = find_num_beats_per_window(p1, p2,
                                                         p3, p4, p5, p6)

    assert a1 == 2
    assert a2 == 4
    assert a3 == 5
    assert a4 == 1
    assert a5 == 5
    assert a6 == 6


def test_findtimepeaks():
    testdata = np.loadtxt("datatestprocess.csv", delimiter=',')

    w = 63
    dw1 = testdata[0:w, :]
    dw2 = testdata[w:w * 2, :]
    dw3 = testdata[w * 2:w * 3, :]
    dw4 = testdata[w * 3:w * 4, :]
    dw5 = testdata[w * 4:w * 5, :]
    dw6 = testdata[w * 5:, :]

    (p1, p2, p3, p4, p5, p6) = find_peaks(dw1, dw2, dw3, dw4, dw5, dw6)

    time_peaks = find_time_beats(dw1,
                                 dw2, dw3, dw4, dw5, dw6,
                                 p1, p2, p3, p4, p5, p6)

    np.testing.assert_array_almost_equal(time_peaks,
                                         ([1.583333, 7.833333, 14.166667,
                                           20.416667, 26.716667, 33.016667]))


def test_find_totalnumbeats():
    a1 = 5
    a2 = 5
    a3 = 6
    a4 = 7
    a5 = 8
    a6 = 9

    numbeats = find_total_numbeats(a1, a2, a3, a4, a5, a6)

    assert numbeats == 40


def test_findbpm():
    duration = 6000
    numbeats = 100
    bpm = find_bpm(duration, numbeats)
    assert bpm == 1


def test_dictionary():
    a = 5
    b = ([4, 3, 2])
    c = 3
    d = 'k'
    e = 1

    dict = createdictionary(a, b, c, d, e)

    assert dict == {'mean_hr_bpm': 5,
                    'voltage_extremes': ([4, 3, 2]),
                    'duration': 3,
                    'num_beats': 'k',
                    'beats': 1}
