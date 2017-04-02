from ftplib import FTP
import os
import sys
import subprocess

server_name = "nomads.ncdc.noaa.gov"  # Server from which to pull the data
gfs_loc = "GFS/analysis_only/"  # location on server of GFS data
host_name = "zbutler@datalab-11.ics.uci.edu"
remote_dir = "/extra/zbutler0/data/gfs/"  # Where we will store raw data
temp_fi_name = "./tmp_2015.grb"  # Place to store grib files locally temporarily
out_temp_arr = "./data/gfs_temp_2015.pkl"  # Place to store temperature tensor
out_hum_arr = "./data/gfs_hum_2015.pkl"  # Place to store temperature tensor


def get_gfs_data(year, partial_data_acquired=False, local=False):
    """ Read GFS
    :param year: Year to get data from
    :param partial_data_acquired: whether or not we have partial data already on our server
    :param local: if true, store locally rather than remotely
    :return:
    """
    days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    day = 1
    month = 1
    day_of_year = 0
    bad_days = 0

    if partial_data_acquired:
        print "partial data yoooo\n\n"
        if local:
            existing_grbs = os.listdir(remote_dir)
        else:
            ls = subprocess.Popen(['ssh',host_name,
                                   'ls %s' % remote_dir],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            out, err = ls.communicate()
            existing_grbs = out.split("\n")
    ftp = FTP(server_name)
    ftp.login("anonymous", "zbutler@fire_prediction.github")
    ftp.cwd(gfs_loc)

    while month < 13:  # Get data for every day of the year
        ym_str = "%d%.2d" % (year, month)
        ymd_str = "%d%.2d%.2d" % (year, month, day)

        foundit = 0
        if partial_data_acquired and (ymd_str + ".grb") in existing_grbs:
            print "passing month %d day %d" % (month, day)
        elif ymd_str not in map(lambda x: x.split("/")[-1], ftp.nlst(ym_str)):
            print "month %d day %d not on server" % (month, day)
            bad_days += 1
        else:
            dir_list_with_fluff = ftp.nlst('/'.join([ym_str, ymd_str]))
            dir_list = map(lambda x: x.split('/')[-1], dir_list_with_fluff)
            # pull the file from the FTP server
            #if ("gfsanl_3_%s_1200_000.grb" % ymd_str) in dir_list:
            #    with open(temp_fi_name, "w") as ftmp:
            #        ftp.retrbinary("RETR %s/%s/gfsanl_3_%s_1200_000.grb" % (ym_str, ymd_str, ymd_str), ftmp.write)
            #    print "Found month %d, day %d (grb)" % (month, day)
            #    foundit = 1
            if ("gfsanl_4_%s_1200_000.grb2" % ymd_str) in dir_list:
                with open(temp_fi_name, "w") as ftmp:
                    ftp.retrbinary("RETR %s/%s/gfsanl_4_%s_1200_000.grb2" % (ym_str, ymd_str, ymd_str), ftmp.write)
                print "Found month %d, day %d (grb2)" % (month, day)
                foundit = 1
            else:
                print "Didn't find month %d, day %d" % (month, day)
                bad_days += 1

            if foundit:
                if local:
                    os.rename(temp_fi_name, remote_dir + ymd_str + ".grb")
                else:
                    os.system("scp %s %s:%s/%s.grb" % (temp_fi_name, host_name, remote_dir, ymd_str))

        if day == days_arr[month-1]:
            day = 1
            month += 1
        else:
            day += 1
        day_of_year += 1

    ftp.quit()
    print "%d bad days" % bad_days


if __name__ == "__main__":
    if sys.argv[2].startswith('f') or sys.argv[2].startswith('F')\
            or sys.argv[2].startswith('N') or sys.argv[2].startswith('n'):
        arg2 = False
    else:
        arg2 = True
    if sys.argv[3].startswith('f') or sys.argv[3].startswith('F')\
            or sys.argv[3].startswith('N') or sys.argv[3].startswith('n'):
        arg3 = False
    else:
        arg3 = True
    get_gfs_data(int(sys.argv[1]), arg2, arg3)