def readinfile(file):

    import numpy as np
    try:
        csv_data = np.loadtxt(file, delimiter= ',')

    except ValueError:
        raise ValueError("Can only input numbers")

    return csv_data

