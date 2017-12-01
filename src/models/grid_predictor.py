"""
Converts model output to a grid.
"""
import numpy as np

from base.model import Model

import datetime as dt

import helper.date_util as du
import helper.df_util as dfu
import helper.geometry as geo

class GridPredictorModel(Model):
    def __init__(self, model, fire_season=((5,14), (8,31))):
        """
        :param model: model to change prediction of
        """
        super(GridPredictorModel, self).__init__()

        self.model = model
        self.fire_season = fire_season

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def pred_to_grid(self, preds, info):
        # Create grid
        year = info.year.iloc[0]

        # TODO: add dynamic sizing (bounds and increment)
        bb = geo.LatLonBoundingBox(55, 71, -165, -138)
        spatial_size = np.shape(bb.make_grid()[0])
        dates = list(du.daterange(dt.date(year, self.fire_season[0][0], self.fire_season[0][1]),
            dt.date(year, self.fire_season[1][0], self.fire_season[1][1]) + du.INC_ONE_DAY))
        grid = np.zeros(spatial_size + (len(dates),))

        # Assign each detection to a cell in time and space
        for pred, row in zip(preds,info.itertuples()):
            lat,lon,date = row.lat_centroid, row.lon_centroid, row.date_local 
            lat_ind,lon_ind = bb.latlon_to_indices(lat,lon,spatial_size[0]) # TODO: add dynamic sizing
            date_ind = (date-dates[0]).days

            grid[lat_ind,lon_ind,date_ind] += pred

        return grid

    def predict(self, X):
        pred = self.model.predict(X) 
        return self.pred_to_grid(pred, X)
