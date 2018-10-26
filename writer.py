import json
import logging

logging.basicConfig(filename="HRMLogging.txt",
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)


def writefile(metrics, file):
    """write dict into file using json

    Args:
         metrics (dict): dictionary from processor with data
         file (json): name of file to write to

    Returns:
        file (JSON file): returns file with desired name
    """
    with open((file + '.json'), 'w') as filename:
        json.dump(metrics, filename)
