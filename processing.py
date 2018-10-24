def find_min_max(data):
    v=[i[1] for i in data]
    extremes = (min(v), max(v))

    return extremes


def find_duration(data):
    t=[i[0] for i in data]
    duration= t[len(t)-1]-t[0]
    return duration


def find_peaks(x):
    import peakutils
    voltage= [i[1] for i in x]

    peakindex = peakutils.indexes(voltage, thres=.5 * max(voltage))

    return peakindex

def find_num_beats(x):


    if len(x) == 0:
        raise TypeError("Heart not beating")

    try:
        numbeats = len(x)
    except TypeError:
        print("Heart not beating")

    return numbeats

def find_time_beats(data,peakindex):

    peak_times = data[peakindex,0]

    return peak_times

def find_bpm(duration,numbeats):

    dur = duration / 60
    bpm = numbeats / dur

    return bpm


def createdictionary(a, b, c, d, e):
    dict={}
    dict["mean_hr_bpm"] = a
    dict["voltage_extremes"] = b
    dict["duration"] = c
    dict["num_beats"] = d
    dict["beats"] = e
    return dict



