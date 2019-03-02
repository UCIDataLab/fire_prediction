"""
Print each message contained in a grib file.
"""

import gribapi
import click
import logging
import pandas
from collections import defaultdict


def ls_grib(src_path, keys):
    keys = ['name', 'level', 'typeOfLevel'] + keys

    messages = defaultdict(list)
    with open(src_path, 'rb') as fin:
        while True:
            gid = gribapi.grib_new_from_file(fin)
            if gid is None:
                break

            for k in keys:
                try:
                    val = gribapi.grib_get(gid, k)
                    messages[k].append(val)
                except Exception as e:
                    logging.error('Failed to get key (%s). Either key is not availabe for the message or it is an '
                                  'array type.' % k)

    return messages, keys


@click.command()
@click.argument('src_path', type=click.Path(exists=True))
@click.option('--keys', default='', type=str)
@click.option('--log', default='INFO')
def main(src_path, keys, log):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info('Reading data from %s' % src_path)

    keys = filter(None, str(keys).split(','))

    messages, keys = ls_grib(src_path, keys)

    df = pandas.DataFrame(messages, columns=keys)
    df.sort_values(keys, inplace=True)

    print(df.to_csv())


if __name__ == '__main__':
    main()
