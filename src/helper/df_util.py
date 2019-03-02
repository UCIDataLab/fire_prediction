"""
Helper functions for dealing with Pandas data frames.
"""

import numpy as np

from helper import date_util as du


def get_year_range(df, col_name):
    """
    """
    try:
        return int(np.min(df[col_name].dt.year)), int(np.max(df[col_name].dt.year))
    except Exception as e:
        return int(np.min(df[col_name]).year), int(np.max(df[col_name]).year)


def add_date_local(df):
    return df.assign(date_local=map(lambda x: du.utc_to_local_time(x[0], x[1], du.round_to_nearest_quarter_hour).date(),
                                    zip(df.datetime_utc, df.lon)))
