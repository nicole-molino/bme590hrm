import numpy
import logging

logging.basicConfig(filename="HRMLogging.txt",
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

def find_min_max(data):
    v = [i[1] for i in data]
    extremes = (min(v), max(v))
    logging.info('Calculated extremes: %s', extremes)
    return extremes


def find_duration(data):
    t = [i[0] for i in data]
    duration = t[len(t) - 1] - t[0]
    logging.info('Calculated duration: %s', duration)
    return duration


def find_windowsize(data):
    time = [i[0] for i in data]
    voltage = [i[1] for i in data]

    if len(time) != len(voltage):
        total_index_data = len(voltage)
    else:
        total_index_data = min(len(time), len(voltage))

    windowsize = round(total_index_data / 6)

    return windowsize


def define_windows(w, data):
    data_w1 = data[0:w, :]
    data_w2 = data[w:w * 2, :]
    data_w3 = data[w * 2:w * 3, :]
    data_w4 = data[w * 3:w * 4, :]
    data_w5 = data[w * 4:w * 5, :]
    data_w6 = data[w * 5:, :]

    return data_w1, data_w2, data_w3, data_w4, data_w5, data_w6


def find_peaks(dw1, dw2, dw3, dw4, dw5, dw6):
    import peakutils

    v1 = [i[1] for i in dw1]
    v2 = [i[1] for i in dw2]
    v3 = [i[1] for i in dw3]
    v4 = [i[1] for i in dw4]
    v5 = [i[1] for i in dw5]
    v6 = [i[1] for i in dw6]

    pi1 = peakutils.indexes(v1, thres=.5 * max(v1))
    pi2 = peakutils.indexes(v2, thres=.5 * max(v2))
    pi3 = peakutils.indexes(v3, thres=.5 * max(v3))
    pi4 = peakutils.indexes(v4, thres=.5 * max(v4))
    pi5 = peakutils.indexes(v5, thres=.5 * max(v5))
    pi6 = peakutils.indexes(v6, thres=.5 * max(v6))

    return pi1, pi2, pi3, pi4, pi5, pi6


def find_num_beats_per_window(p1, p2, p3, p4, p5, p6):
    return (len(p1), len(p2), len(p3), len(p4), len(p5), len(p6))


def find_time_beats(dw1, dw2, dw3, dw4, dw5, dw6, p1, p2, p3, p4, p5, p6):
    time_beats = numpy.concatenate((dw1[p1, 0], dw2[p2, 0],
                                    dw3[p3, 0], dw4[p4, 0],
                                    dw5[p5, 0], dw6[p6, 0]))

    logging.info('Calculated time of beats: %s', time_beats)
    return time_beats


def find_total_numbeats(a1, a2, a3, a4, a5, a6):
    numbeats = a1 + a2 + a3 + a4 + a5 + a6


    logging.info('Calculated total number of beats: %s', numbeats)
    return numbeats


def find_bpm(duration, numbeats):
    dur = duration / 60
    bpm = numbeats / dur
    print('bpm calculated')


    if bpm > 300:
        raise TypeError("TOOOOOOO HIGH")


      #  except ValueError:
          #  pass
    #else:
       # logging.info('Calculated BPM: %s', bpm)

    return bpm


def createdictionary(a, b, c, d, e):
    dict = {}
    dict["mean_hr_bpm"] = a
    dict["voltage_extremes"] = b
    dict["duration"] = c
    dict["num_beats"] = d
    dict["beats"] = e
    return dict
