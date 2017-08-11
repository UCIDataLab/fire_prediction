"""
Useful functions for loading interim and processed data.
"""

import cPickle as pickle

def load_modis_df(src):
    with open(src, 'rb') as fin:
        return pickle.load(fin)
