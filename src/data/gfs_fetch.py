"""
Fetches GFS (Global Forecasting System) data.
"""

import click
import logging
import os
import sys

class GfsFetch(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def fetch(self):
        raise NotImplementedError()
        logging.info('Starting fetch for GFS')
        logging.info('End fetch for GFS')


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
def main(data_dir):
    logging.info("Storing data in %s" % data_dir)

    GfsFetch(data_dir).fetch()

if __name__=='__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
