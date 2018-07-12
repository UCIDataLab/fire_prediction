import numpy as np
import luigi
import datetime as dt
import pandas as pd
import os
import gzip

from pipeline import FtpFile, change_data_dir_path
from data.gfs_choices import GFS_BOUNDING_BOXES
from helper.geometry import filter_bounding_box_df

MODIS_SERVER_NAME = 'fuoco.geog.umd.edu'
MODIS_SERVER_USERNAME = 'fire'
MODIS_SERVER_PASSWORD = 'burnt'
MODIS_SERVER_DATA_DIR = 'modis/C6/mcd14ml'

def build_modis_server_file_path(data_dir, date):
    fn = 'MCD14ML.%s.006.01.txt.gz' % date.strftime('%Y%m')
    return os.path.join(data_dir, fn)

class ModisFtpFileDownload(luigi.Task):
    server_name = luigi.parameter.Parameter(default=MODIS_SERVER_NAME)
    server_username = luigi.parameter.Parameter(default=MODIS_SERVER_USERNAME, significant=False)
    server_password = luigi.parameter.Parameter(default=MODIS_SERVER_PASSWORD, significant=False)
    server_data_dir = luigi.parameter.Parameter(MODIS_SERVER_DATA_DIR)

    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default='modis/raw')

    month_sel = luigi.parameter.MonthParameter()

    resources = {'ftp': 1}

    def requires(self):
        file_path = build_modis_server_file_path(self.server_data_dir, self.month_sel)
        return FtpFile(file_path=file_path)

    def run(self):
        # Copy ftp file from server to local dest
        with self.output().temporary_path() as temp_output_path:
            self.input().get(temp_output_path)

    def output(self):
        dest_path = change_data_dir_path(self.server_data_dir, 
                os.path.join(self.data_dir, self.dest_data_dir), self.input().path)
        return luigi.LocalTarget(dest_path)

def monthrange(start_date, end_date):
    """Iterate through months in date range (inclusive)."""

    cur_month, cur_year = start_date.month, start_date.year
    end_month, end_year = end_date.month, end_date.year

    while (cur_month != end_month) or (cur_year != end_year):
        yield dt.date(cur_year, cur_month, 1)

        cur_year += cur_month == 12
        cur_month = np.remainder(cur_month, 12) + 1

    yield dt.date(end_year, end_month, 1)

class ModisAggregate(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default='interim/modis/aggregated')

    start_month_sel = luigi.parameter.MonthParameter()
    end_month_sel = luigi.parameter.MonthParameter()

    def requires(self):
        months = monthrange(self.start_month_sel, self.end_month_sel)
        return [ModisFtpFileDownload(data_dir=self.data_dir, month_sel=month) for month in months]

    def run(self):
        frames = []

        for file_path in [in_f.path for in_f in self.input()]:
            with gzip.open(file_path, 'rb') as fin:
                df = pd.read_csv(fin, sep=' ', skipinitialspace=True, index_col=None, 
                        parse_dates=[['YYYYMMDD', 'HHMM']], infer_datetime_format=True)

                # Rename datetime column
                df.rename(columns = {'YYYYMMDD_HHMM': 'datetime_utc'}, inplace=True)

                #df.index.names = ['datetime_utc'] # Make index the UTC datetime
                frames.append(df)

        df = pd.concat(frames)

        with self.output().temporary_path() as temp_output_path:
            df.to_pickle(temp_output_path)

    def output(self):
        date_fmt = '%Y%m'

        start_month_str = self.start_month_sel.strftime(date_fmt)
        end_month_str = self.end_month_sel.strftime(date_fmt)

        fn = 'mcd14ml_006_%s_%s.pkl' % (start_month_str, end_month_str)
        dest_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

        return luigi.LocalTarget(dest_path)

class ModisFilterRegion(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default='interim/modis/region')

    start_month_sel = luigi.parameter.MonthParameter()
    end_month_sel = luigi.parameter.MonthParameter()

    bounding_box_sel_name = luigi.parameter.Parameter(default='alaska')

    def requires(self):
        return ModisAggregate(data_dir=self.data_dir, start_month_sel=self.start_month_sel, end_month_sel=self.end_month_sel)

    def run(self):
        df = pd.read_pickle(self.input().path)

        df = df[df['type']==0] # Include only vegetation fires

        bounding_box = GFS_BOUNDING_BOXES[self.bounding_box_sel_name]
        df = filter_bounding_box_df(df, bounding_box) # Only use fires in bounding box

        # Localize datetime to UTC
        df.datetime_utc = df.datetime_utc.dt.tz_localize('utc')

        # Reset index numbering after dropping rows
        df.reset_index(drop=True, inplace=True)

        with self.output().temporary_path() as temp_output_path:
            df.to_pickle(temp_output_path)

    def output(self):
        dirname, fn = os.path.split(self.input().path)
        fn, _ = os.path.splitext(fn)
        fn = fn + '_%s' % self.bounding_box_sel_name + '.pkl'
        dest_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

        return luigi.LocalTarget(dest_path)

