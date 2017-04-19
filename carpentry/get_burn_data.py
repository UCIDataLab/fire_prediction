from ftplib import FTP
from util.tile2latlon import tile2latlon
import ftplib
import os
import sys
import pandas as pd

server_name = "fuoco.geog.umd.edu"
base_server_dir = "db/MCD64A1/"
out_dir = "data/burnt/modis/"


def get_burn_data(h_range, v_range, out_dir=out_dir):
    """ Read MODIS burn data
    :param year: Year to get data from
    :param partial_data_acquired: whether or not we have partial data already on our server
    :param local: if true, store locally rather than remotely
    :return:
    """
    ftp = FTP(server_name, timeout=7200)
    ftp.login("fire", "burnt")
    ftp.cwd(base_server_dir)

    for h in h_range:
        for v in v_range:
            dir_str = "h%.2dv%.2d/" % (h, v)
            try:
                filenames = ftp.nlst(dir_str)
            except ftplib.error_perm:
                print "no dir " + dir_str
                continue
            if not os.path.exists(out_dir + dir_str):
                os.mkdir(out_dir + dir_str)
            for filename in filenames:
                with open(out_dir + dir_str + ".".join(filename.split(".")[1:]), 'w') as fout:
                    ftp.retrbinary("RETR %s" % (dir_str + filename), fout.write)
            print "Finished h %d v %d" % (h, v)
    ftp.quit()


if __name__ == "__main__":
    h_range = range(int(sys.argv[1].split(",")[0]), int(sys.argv[1].strip().split(",")[1]))
    v_range = range(int(sys.argv[2].split(",")[0]), int(sys.argv[2].strip().split(",")[1]))
    get_burn_data(h_range, v_range)
