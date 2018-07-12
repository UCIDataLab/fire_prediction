import numpy as np
import luigi

import os
import pandas as pd

from modis_pipeline import ModisFilterRegion
from features.generate_fire_cube import GridGenerator
from data.gfs_choices import GFS_BOUNDING_BOXES

class FireGridGeneration(luigi.Task):
    """ Generate detection or ignition fire grid from detections. """
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default='interim/modis/fire_grid')

    start_date = luigi.parameter.MonthParameter()
    end_date = luigi.parameter.MonthParameter()

    bounding_box_sel_name = luigi.parameter.Parameter(default='alaska')

    def requires(self):
        return ModisFilterRegion(data_dir=self.data_dir, start_month_sel=self.start_date, 
                end_month_sel=self.end_date, bounding_box_sel_name=self.bounding_box_sel_name)

    def run(self):
        # Load input
        df = pd.read_pickle(self.input().path)

        # Build transformer
        bounding_box = GFS_BOUNDING_BOXES[self.bounding_box_sel_name]
        gg = GridGenerator(bounding_box=bounding_box)

        # Transform dataframe to grid
        ds = gg.transform(df)

        with self.output().temporary_path() as temp_output_path:
            encoding = {k: {'zlib': True, 'complevel': 1}  for k in ds.data_vars.keys()}
            ds.to_netcdf(temp_output_path, engine='netcdf4', encoding=encoding)

    def output(self):
        _, in_fn = os.path.split(self.input().path)
        in_fn, _ = os.path.splitext(in_fn)

        date_fmt = '%Y%m%d'
        start_date_str = self.start_date.strftime(date_fmt)
        end_date_str =  self.end_date.strftime(date_fmt)

        fn = 'fire_grid_%s_%s_%s_%s.nc' % (in_fn, self.bounding_box_sel_name, start_date_str, end_date_str)
        file_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

        return luigi.LocalTarget(file_path)


class FireClustering(luigi.Task):
    """ Assign cluster ids to fire events. """
    def requires(self):
        pass

    def run(self):
        pass

    def output(self):
        pass

