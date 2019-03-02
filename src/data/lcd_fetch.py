"""
Fetches station Local Climatological Data (LCD).

Since this data requires using a web interface request system, this code only informs the user what they need to do.
"""

import logging

import click


class LcdFetch(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @staticmethod
    def fetch():
        print(
            'Local Climatological Data (LCD) requires using the online request system at '
            '"https://www.ncdc.noaa.gov/cdo-web/datatools/lcd". Refer to the "Data" section of the "Getting Started" '
            'doc.')


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
def main(data_dir):
    logging.info('Storing data in %s' % data_dir)

    LcdFetch(data_dir).fetch()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
