def readinfile(file):
    """read in csv file and save to space

    Args:
        file (.csv): csv file of voltage and time data

    Returns:
        data (tuple): numpy array containing voltage and time data

    Raises:
        ValueError: if non-numbers in file and breaks code

    """
    import logging

    logging.basicConfig(filename="HRMLogging.txt",
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)

    import numpy as np
    try:
        csv_data = np.loadtxt(file, delimiter=',')

    except ValueError:
        logging.warning("Tried to input non-numbers, process stopped")
        raise ValueError("Can only input numbers")

    return csv_data
