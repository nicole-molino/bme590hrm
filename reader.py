def readinfile(file):
    import csv
    with open(file) as fh:
        csv_reader = csv.reader(fh, delimiter = ',')
        time = []
        voltage = []

        for row in csv_reader:
            t = float(row[0])
            v = float(row[1])
            time.append(t)
            voltage.append(v)
        return time, voltage