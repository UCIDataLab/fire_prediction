"""
Preform processing on individual data source (e.g. clustering on MODIS).
"""

import click
import logging

@click.command()
@click.argument('src_dir', type=click.Path(exists=True))
def main(data_dir):
    """
    Calls each processing script.
    """
    pass

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()








