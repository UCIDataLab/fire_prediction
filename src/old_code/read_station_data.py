import sys
from datetime import datetime

import cPickle
import numpy as np
import pandas as pd
from util.daymonth import monthday2day

fields_of_interest = dict()
fields_of_interest["HOURLYDRYBULBTEMPC"] = "temp"
fields_of_interest["HOURLYWindSpeed"] = "wind"
fields_of_interest["HOURLYRelativeHumidity"] = "humidity"
fields_of_interest["DAILYPrecip"] = "rain"
# fields_of_interest["LATITUDE"] = "lat"
# fields_of_interest["LONGITUDE"] = "lon"
fields_of_interest["DATE"] = "date"
DT_FMT = "%H:%M"  # Format for datetime objects
NOON = datetime.strptime("12:00", DT_FMT)


def station_csv_to_pandas(csv_file, outfi):
    with open(csv_file) as fcsv:
        header = fcsv.readline().strip().split(",")
        name2col_num = dict()
        for key, val in fields_of_interest.iteritems():
            name2col_num[val] = header.index(key)
        res_dict = dict()
        for key in name2col_num.keys():
            if key == "date":
                res_dict["day"] = []
                res_dict["month"] = []
                res_dict["year"] = []
                res_dict["dayofyear"] = []
                continue
            res_dict[key] = []

        # first_time = 1
        prev_spl = []
        prev_hour = 23
        prev_day = 1
        prev_time = datetime.strptime("23:00", DT_FMT)
        for line in fcsv:
            spl = line.strip().split(",")
            # if first_time:
            #    res_dict['lat'] = float(spl[name2col_num['lat']])
            #    res_dict['lon'] = float(spl[name2col_num['lon']])
            #    first_time = 0
            dt = spl[name2col_num["date"]]
            year = int(dt.split(" ")[0].split("-")[0])
            month = int(dt.split(" ")[0].split("-")[1])
            day = int(dt.split(" ")[0].split("-")[2])
            hour = int(dt.split(" ")[1].split(":")[0])
            minute = int(dt.split(" ")[1].split(":")[1])
            if prev_spl:
                print
                prev_spl[name2col_num["rain"]]
            if hour >= 12 and prev_hour == 11:
                prev_delta = NOON - prev_time
                current_delta = datetime.strptime("%d:%d" % (hour, minute), DT_FMT) - NOON
                if abs(prev_delta.total_seconds()) < abs(current_delta.total_seconds()):
                    my_spl = prev_spl
                else:
                    my_spl = spl
                for key in name2col_num.keys():
                    if key == "date":
                        res_dict["day"].append(day)
                        res_dict["month"].append(month)
                        res_dict["year"].append(year)
                        res_dict["dayofyear"].append(monthday2day(month, day, year % 4 == 0))
                        continue
                    if key == "lat" or key == "lon" or key == "rain":
                        continue
                    try:
                        res_dict[key].append(float(my_spl[name2col_num[key]]))
                    except ValueError:
                        res_dict[key].append(np.nan)
            elif prev_day != day:  # the last timestamp was the last of that day and thus has rain
                rain_str = prev_spl[name2col_num["rain"]]
                if rain_str.startswith("T"):
                    res_dict["rain"].append(0.0)
                else:
                    res_dict["rain"].append(float(rain_str))

            prev_day = day
            prev_hour = hour
            prev_time = datetime.strptime("%d:%d" % (hour, minute), DT_FMT)
            prev_spl = spl
    # Gotta do rain one last time:
    rain_str = prev_spl[name2col_num["rain"]]
    if rain_str.startswith("T"):
        res_dict["rain"].append(0.0)
    else:
        res_dict["rain"].append(float(rain_str))

    for key, val in res_dict.iteritems():
        print
        "%s: %d" % (key, len(val))
    res_df = pd.DataFrame(res_dict)
    with open(outfi, 'w') as fout:
        cPickle.dump(res_df, fout, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    station_csv_to_pandas(sys.argv[1], sys.argv[2])
