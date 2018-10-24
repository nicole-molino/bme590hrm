def readinfile(file):
    import numpy as np
    try:
        csv_data = np.loadtxt(file, delimiter= ',')

    except ValueError:
        print("hey")

    return csv_data

