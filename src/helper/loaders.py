"""
Useful functions for loading interim and processed data.
"""
import pickle
import pandas as pd

def load_pickle(src):
    return pd.read_pickle(src)

