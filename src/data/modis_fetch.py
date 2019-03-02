"""
Fetches MODIS (Moderate Resolution Imaging Spectroradiometer) data.
"""

import logging

import click


class ModisFetch(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def fetch(self):
        raise NotImplementedError()
        logging.info('Starting fetch for MODIS')
        logging.info('End fetch for MODIS')


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
def main(data_dir):
    logging.info("Storing data in %s" % data_dir)

    ModisFetch(data_dir).fetch()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
