"""
Fetch raw data from external sources then extract and aggregate it. Raw data includes MODIS and GFS (station LCD data requires using a web interface).
"""

import click
import logging

from gfs_fetch import GfsFetch
from modis_fetch import ModisFetch
from lcd_fetch import LcdFetch

fetch_classes = [GfsFetch, ModisFetch, LcdFetch]

@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
def main(data_dir):
    """
    Calls each data fetch script to populate data/raw.
    """
    logging.info("Storing data in %s" % data_dir)

    for fetch_class in fetch_classes:
        fetch_class(data_dir).fetch()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()








