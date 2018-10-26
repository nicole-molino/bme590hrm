if __name__ == "__main__":
    import numpy as np
    import logging

    file= "test_data/test_data12.csv"
    data = np.loadtxt(file, delimiter=',')

    logging.basicConfig(filename="HRMLogging.txt",
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)

    logging.info('File: %s', file)

    import processing as p

    voltage_extremes = p.find_min_max(data)
    # print(voltage_extremes)
    duration = p.find_duration(data)
    # print(duration)
    windowsize = p.find_windowsize(data)
    # print(windowsize)
    (dw1, dw2, dw3, dw4, dw5, dw6) = p.define_windows(windowsize, data)
    (pi1, pi2, pi3, pi4, pi5, pi6) = p.find_peaks(dw1,
                                                  dw2, dw3, dw4, dw5, dw6)
    (nb1, nb2, nb3, nb4, nb5, nb6) = p.find_num_beats_per_window(pi1,
                                                                 pi2, pi3, pi4,
                                                                 pi5, pi6)
    # print(nb1, nb2, nb3, nb4, nb5, nb6)
    time_beats = p.find_time_beats(dw1, dw2, dw3, dw4,
                                   dw5, dw6, pi1, pi2,
                                   pi3, pi4, pi5, pi6)

    numbeats = p.find_total_numbeats(nb1, nb2, nb3, nb4, nb5, nb6)



    # print(numbeats)
    try:
        bpm = p.find_bpm(duration, numbeats)
    except TypeError:
        logging.warning("TOO TOO HIGH")
        bpm = p.find_bpm(duration, numbeats)

    # print(bpm)
    metrics = p.createdictionary(bpm, voltage_extremes,
                                 duration, numbeats, time_beats)
    print(metrics)
