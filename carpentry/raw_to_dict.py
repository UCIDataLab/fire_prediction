import cPickle
import matplotlib.pyplot as plt


def read_raw_data(instr, outstr=None):
    with open(instr) as fin:
        data_list = []
        data_list.append("Format: year,month,day,lat,long,frp,conf")
        fin.readline()
        for line in fin:
            yyyymmdd = line.split()[0]
            year = int(yyyymmdd[0:4])
            month = int(yyyymmdd[4:6])
            day = int(yyyymmdd[6:])
            lat = float(line.split()[3])
            lon = float(line.split()[4])
            frp = float(line.split()[8])
            conf = float(line.split()[9]) / 100.

            data_list.append([year, month, day, lat, lon, frp, conf])
    return data_list


def plot_lat_longs(data_list):
    lat_arr = map(lambda x: x[3], data_list[1:])
    long_arr = map(lambda x: x[4], data_list[1:])
    plt.plot(long_arr, lat_arr, 'r+')
    plt.show()
