
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

def validate(x):
    import numpy

    time = x[0]
    voltage = x[1]
    Ltime = len(time)
    Lvolt = len(voltage)


    #if Ltime != Lvolt:

    for number in time:
        if number < 0:
            print('Time value less than 0')

def findpeaks(x):
    import numpy
    import peakutils
    time=x[0]
    voltage=x[1]
    hh = numpy.array(voltage)

    peakindex = peakutils.indexes(hh, thres=.4, min_dist=0.5)


    return peakindex

def find_min_max(data):
    v=data[1]
    extremes = (min(v), max(v))
    print("minmax tuple=", extremes)

    return extremes

def find_duration(data):
    t=data[0]
    duration= t[len(t)-1]-t[0]
    print("duration=", duration)
    return duration

def createdictionary(x,y):
    dict={}
    dict["voltage_extremes"]=x
    dict["duration"]=y
    return dict

if __name__ == "__main__":
    data = readinfile('test_data/test_data2.csv')
    print("all data=", data)
    validate(data)
    extremes = find_min_max(data)
    duration = find_duration(data)

    metrics=createdictionary(extremes, duration)
    print(metrics)
