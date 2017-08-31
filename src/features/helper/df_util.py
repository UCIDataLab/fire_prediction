"""
Helper functions for dealing with Pandas data frames.
"""

import numpy as np

def get_year_range(df, col_name):
    """
    Assumes index is the datetime.
    """
    return int(np.min(df[col_name].dt.year)), int(np.max(df[col_name].dt.year))
