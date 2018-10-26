import numpy
import logging

logging.basicConfig(filename="HRMLogging.txt",
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

def find_min_max(data):
    """ finds extremes of voltage

    Args:
        data (ndarray): time and voltage

    Returns:
        extremes (list): min and max of voltage


    """
    v = [i[1] for i in data]
    extremes = [min(v), max(v)]
    logging.info('Calculated extremes: %s', extremes)
    return extremes


def find_duration(data):
    """ calculate duration of data in seconds

    Args:
         data (ndarray): time and voltage

    Returns:
        duration (float): duration in seconds


    """
    t = [i[0] for i in data]
    duration = t[len(t) - 1] - t[0]
    logging.info('Calculated duration: %s', duration)
    return duration


def find_windowsize(data):
    """divides the trial into 6 windows and finds size of each window

    Args:
        data (ndarray): time and voltage

    Returns:
        windowsize (int): index size for one sixth of trial


    """
    time = [i[0] for i in data]
    voltage = [i[1] for i in data]

    if len(time) != len(voltage):
        total_index_data = len(voltage)
    else:
        total_index_data = min(len(time), len(voltage))

    windowsize = round(total_index_data / 6)

    return windowsize


def define_windows(w, data):
    """based on window size, gives indices for each of 6 windows

    Args:
        data (ndarray): time and voltage
        w (int): index size for one sixth of trial

    Returns:
        dw1 (ndarray) : voltage and time data for window 1
        dw2 (ndarray) : voltage and time data for window 2
        dw3 (ndarray) : voltage and time data for window 3
        dw4 (ndarray) : voltage and time data for window 4
        dw5 (ndarray) : voltage and time data for window 6
        dw6 (ndarray) : voltage and time data for window 6


    """
    data_w1 = data[0:w, :]
    data_w2 = data[w:w * 2, :]
    data_w3 = data[w * 2:w * 3, :]
    data_w4 = data[w * 3:w * 4, :]
    data_w5 = data[w * 4:w * 5, :]
    data_w6 = data[w * 5:, :]

    return data_w1, data_w2, data_w3, data_w4, data_w5, data_w6


def find_peaks(dw1, dw2, dw3, dw4, dw5, dw6):
    """find peaks in data (beats) for each window using peakutils

     Args:
        dw1 (ndarray) : voltage and time data for window 1
        dw2 (ndarray) : voltage and time data for window 2
        dw3 (ndarray) : voltage and time data for window 3
        dw4 (ndarray) : voltage and time data for window 4
        dw5 (ndarray) : voltage and time data for window 6
        dw6 (ndarray) : voltage and time data for window 6

    Returns:
        pi1 (ndarray) : index of each beat in window 1
        pi2 (ndarray) : index of each beat in window 2
        pi3 (ndarray) : index of each beat in window 3
        pi4...
        pi5...
        pi6...


         """
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
    """calculate number of beats per window

    Args:
        pi1 (ndarray) : index of each beat in window 1
        pi2 (ndarray) : index of each beat in window 2
        pi3 (ndarray) : index of each beat in window 3
        pi4...
        pi5...
        pi6...

    Returns:
        nb1 (int): number of beats found in window 1
        nb2 (int): number of beats found in window 2
        nb3 (int): number of beats found in window 3
        nb4 (int): ...
        nb5 (int): ...
        nb6 (int): ...
        """


    return (len(p1), len(p2), len(p3), len(p4), len(p5), len(p6))


def find_time_beats(dw1, dw2, dw3, dw4, dw5, dw6, p1, p2, p3, p4, p5, p6):
    """calculate time when beat beat occurs

    Args:
        dw1 (ndarray) : voltage and time data for window 1
        dw2 (ndarray) : voltage and time data for window 2
        dw3 (ndarray) : voltage and time data for window 3
        dw4 (ndarray) : voltage and time data for window 4
        dw5 (ndarray) : voltage and time data for window 6
        dw6 (ndarray) : voltage and time data for window 6
        pi1 (ndarray) : index of each beat in window 1
        pi2 (ndarray) : index of each beat in window 2
        pi3 (ndarray) : index of each beat in window 3
        pi4...
        pi5...
        pi6...

    Returns:
        time_beats (tuple): time each beat occurred

    """
    time_beats = numpy.concatenate((dw1[p1, 0], dw2[p2, 0],
                                    dw3[p3, 0], dw4[p4, 0],
                                    dw5[p5, 0], dw6[p6, 0]))

    logging.info('Calculated time of beats: %s', time_beats)
    return time_beats


def find_total_numbeats(nb1, nb2, nb3, nb4, nb5, nb6):
    """" add up beats from each window

    Args:
        nb1 (int): number of beats found in window 1
        nb2 (int): number of beats found in window 2
        nb3 (int): number of beats found in window 3
        nb4 (int): ...
        nb5 (int): ...
        nb6 (int): ...

    Returns:
        numbeats (int): total number of beats during trials

    """
    numbeats = nb1 + nb2 + nb3 + nb4 + nb5 + nb6


    logging.info('Calculated total number of beats: %s', numbeats)
    return numbeats


def find_bpm(duration, numbeats):
    """calculate HR in beats per minute

    Args:
        duration (int) : total time of trial
        numbeats (int): total number of beats during trials

    Returns:
        bpm (int) : heart rate

    Raises:
        TypeError: if the heart rate is above 300 bpm


    """
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


def createdictionary(bpm, extremes, duration, numbeats, time_beats):
    """creates dictionary to save data

    Args:
         bpm (int) : heart rate
         extremes (list): min and max of voltage
         duration (int) : time of trial
         numbeats (int): total number of beats during trials
         time_beats (tuple): time each beat occurred

    Returns:
        metrics (dict): dictionary with saved necessary data

    """
    dict = {}
    dict["mean_hr_bpm"] = bpm
    dict["voltage_extremes"] = extremes
    dict["duration"] = duration
    dict["num_beats"] = numbeats
    dict["beats"] = time_beats
    return dict
