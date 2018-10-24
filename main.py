if __name__ == "__main__":











    import numpy as np

    data = np.loadtxt("datatestprocess.csv", delimiter=',')


    from processing import find_min_max
    from processing import find_duration
    from processing import find_peaks
    from processing import find_num_beats
    from processing import find_time_beats
    from processing import find_bpm
    from processing import createdictionary

    voltage_extremes = find_min_max(data)
    duration = find_duration(data)
    peakindex = find_peaks(data)
    num_beats = find_num_beats(peakindex)
    beats = find_time_beats(data,peakindex)
    mean_hr_bpm = find_bpm(duration, num_beats)

    metrics=createdictionary(mean_hr_bpm, voltage_extremes, duration, num_beats, beats)
    print(metrics)