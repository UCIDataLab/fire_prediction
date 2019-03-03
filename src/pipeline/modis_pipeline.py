import datetime as dtime
import gzip
import os

import luigi
import pandas as pd

from src.helper.date_util import date_range_months
from src.helper.geometry import filter_bounding_box_df
from .pipeline_helper import FtpFile, change_data_dir_path
from .pipeline_params import (REGION_BOUNDING_BOXES,
                              MODIS_RAW_DATA_DIR, MODIS_AGGREGATED_DATA_DIR, MODIS_REGION_DATA_DIR,
                              MODIS_SERVER_NAME, MODIS_SERVER_USERNAME, MODIS_SERVER_PASSWORD, MODIS_SERVER_DATA_DIR)


def build_modis_server_file_path(data_dir, date):
    fn = 'MCD14ML.%s.006.01.txt.gz' % date.strftime('%Y%m')
    return os.path.join(data_dir, fn)


class ModisFtpFileDownload(luigi.Task):
    data_dir: str = luigi.parameter.Parameter()
    dest_data_dir: str = luigi.parameter.Parameter(default=MODIS_RAW_DATA_DIR)

    server_name = luigi.parameter.Parameter(default=MODIS_SERVER_NAME)
    server_username = luigi.parameter.Parameter(default=MODIS_SERVER_USERNAME, significant=False)
    server_password = luigi.parameter.Parameter(default=MODIS_SERVER_PASSWORD, significant=False)
    server_data_dir = luigi.parameter.Parameter(MODIS_SERVER_DATA_DIR)

    month_sel: dtime.date = luigi.parameter.MonthParameter()

    resources = {'ftp': 1}

    def requires(self):
        file_path = build_modis_server_file_path(self.server_data_dir, self.month_sel)
        return FtpFile(server_name=self.server_name, server_username=self.server_username,
                       server_password=self.server_password, file_path=file_path)

    def run(self):
        # Copy ftp file from server to local dest
        with self.output().temporary_path() as temp_output_path:
            self.input().get(temp_output_path)

    def output(self):
        dest_path = change_data_dir_path(self.server_data_dir,
                                         os.path.join(self.data_dir, self.dest_data_dir), self.input().path)
        return luigi.LocalTarget(dest_path)


class ModisAggregate(luigi.Task):
    data_dir: str = luigi.parameter.Parameter()
    dest_data_dir: str = luigi.parameter.Parameter(default=MODIS_AGGREGATED_DATA_DIR)

    start_month_sel: dtime.date = luigi.parameter.MonthParameter()
    end_month_sel: dtime.date = luigi.parameter.MonthParameter()

    def requires(self):
        months = date_range_months(self.start_month_sel, self.end_month_sel)
        return [ModisFtpFileDownload(data_dir=self.data_dir, month_sel=month) for month in months]

    def run(self):
        frames = []

        for file_path in [in_f.path for in_f in self.input()]:
            with gzip.open(file_path, 'rb') as fin:
                df = pd.read_csv(fin, sep=' ', skipinitialspace=True, index_col=None,
                                 parse_dates=[['YYYYMMDD', 'HHMM']], infer_datetime_format=True)

                # Rename datetime column
                df.rename(columns={'YYYYMMDD_HHMM': 'datetime_utc'}, inplace=True)

                # df.index.names = ['datetime_utc'] # Make index the UTC datetime
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
    data_dir: str = luigi.parameter.Parameter()
    dest_data_dir: str = luigi.parameter.Parameter(default=MODIS_REGION_DATA_DIR)

    start_month_sel = luigi.parameter.MonthParameter()
    end_month_sel = luigi.parameter.MonthParameter()

    bounding_box_sel_name = luigi.parameter.Parameter(default='alaska')

    def requires(self):
        return ModisAggregate(data_dir=self.data_dir, start_month_sel=self.start_month_sel,
                              end_month_sel=self.end_month_sel)

    def run(self):
        df = pd.read_pickle(self.input().path)

        df = df[df['type'] == 0]  # Include only vegetation fires

        bounding_box = REGION_BOUNDING_BOXES[self.bounding_box_sel_name]
        df = filter_bounding_box_df(df, bounding_box)  # Only use fires in bounding box

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
