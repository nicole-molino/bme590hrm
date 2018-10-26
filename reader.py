def readinfile(file):
    """read in csv file and save to space

    Args:
        file (.csv): csv file of voltage and time data

    Returns:
        data (tuple): numpy array containing voltage and time data

    """

    import numpy as np
    try:
        csv_data = np.loadtxt(file, delimiter=',')

    except ValueError:
        raise ValueError("Can only input numbers")

    return csv_data
