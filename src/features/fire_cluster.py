"""
Converts a data frame of fire data to a cluser data frame.
"""

import click
import cPickle as pickle
import pandas
import logging

from base.converter import Converter

class FireDfToCluserConverter(Converter):
    """
    Converts a data frame of fire data to a cluser data frame.
    """
    def __init__(self):
        super(FireDfToClusterConverter, self).__init__()

    def load(self, src_path):
        logging.info('Loading file from %s' % src_path)
        with open(src_path, 'rb') as fin:
            data = pickle.load(fin)

        return data

    def save(self, dest_path, data):
        logging.info('Saving data frame to %s' % dest_path)
        with open(dest_path, 'wb') as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL

    def transform(self, data):
        logging.debug('Applying transforms to data frame')
        # TODO: Write cluster code
        pass



@click.command()
@click.argument('src_path', type=click.Path(exists=True))
@click.argument('dest_path')
@click.option('--log', default='INFO')
def main(src_dir, dest_path, log):
    """
    Load fire data frame and create clusters.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info('Starting fire data frame to cluster conversion')
    FireDfToClusterConverter().convert(src_path, dest_path)
    logging.info('Finished fire data frame to cluster conversion')


if __name__ == '__main__':
    main()
