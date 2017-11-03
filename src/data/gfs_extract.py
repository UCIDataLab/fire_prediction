import os
import click
import logging
import multiprocessing
import cPickle as pickle

import grib
from helper import date_util as du
from ftp_async import PoolLimitedTaskQueue

year_month_dir_fmt = "%d%.2d"
year_month_day_dir_fmt = "%d%.2d%.2d"
grib_file_fmt_half_deg = "gfsanl_4_%s_%.4d_%.3d.grb2"
grib_file_fmt_one_deg = "gfsanl_3_%s_%.4d_%.3d.grb"

SCALE_HALF_DEG = '4'
SCALE_ONE_DEG = '3'

times = [0, 600, 1200, 1800]
offsets = [0, 3, 6,]
time_offset_list = [(t,o) for t in times for o in offsets]

class Writer(object):
    def __init__(self, write_queue):
        self.write_queue = write_queue

    def write(self):
        while True:
            dest_path, data = self.write_queue.get()
            if data is None:
                logging.debug('Writing queue has reached terminal element, exiting')
                return
            logging.debug('Writing %s, %d items in queue' % (dest_path, self.write_queue.qsize()))
            with open(dest_path, 'wb') as fout:
                pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

def extract_thread(context, src_path, dest_path):
    logging.debug('Extracting %s' % src_path)
    selector = context['selector']
    write_queue = context['write_queue']

    with open(src_path, 'rb') as fin:
        grib_file = grib.GribFile(fin)
        try:
            extracted = selector.select(grib_file)
        except Exception as err:
            logging.debug('Exception "%s" encountered while selecting from file "%s"' % (str(err), src_path))
            extracted = None

    if extracted:
        write_queue.put((dest_path, extracted))


class GfsExtractor(object):
    def __init__(self, pool_size, queue_size):
        self.pool_size = pool_size
        self.queue_size = queue_size

        self.started = False

    def start(self):
        self.write_queue = multiprocessing.Queue(maxsize=self.queue_size)

        self.pool = PoolLimitedTaskQueue(self.pool_size, self.pool_size*2, initializer=self.init_worker)

        self.writer = multiprocessing.Process(target=Writer(self.write_queue).write)
        self.writer.start()

        self.started = True

    def extract(self, src_path, dest_path):
        if not self.started:
            raise Exception('Not started')

        self.pool.apply_async(extract_thread, args=(src_path, dest_path))

    def join(self):
        if not self.started:
            raise Exception('Not started')

        self.pool.close()
        self.pool.join()

        self.write_queue.put((None, None))
        logging.debug('Waiting on writer to finish processing queue')
        self.writer.join()

    def init_worker(self, context):
        selections = grib.get_default_selections()
        bb = grib.get_default_bounding_box()
        context['selector'] = grib.GribSelector(selections, bb)

        context['write_queue'] = self.write_queue


class GfsExtract(object):
    def __init__(self, src_dir, dest_dir, start_year, end_year, scale_sel):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.year_range = (start_year, end_year)

        # Choose file format based on selected scale
        if scale_sel==SCALE_HALF_DEG:
            self.grib_file_fmt = grib_file_fmt_half_deg
        elif scale_sel==SCALE_ONE_DEG:
            self.grib_file_fmt = grib_file_fmt_one_deg
        else:
            raise ValueError('Scale selction "%s" is invalid.' % scale_sel)

        self.extractor = GfsExtractor(pool_size=8, queue_size=50)

    def src_to_dest_path(self, path):
        # Replace src_dir with dest_dir
        path = path.split(self.src_dir)[1]
        path = os.path.join(self.dest_dir, path.lstrip('/'))

        # Replace extension with .pkl
        return os.path.splitext(path)[0] + '.pkl'

    def extract(self):
        # Find all src files to process
        available_files = self.get_available_files()
        logging.debug('Finished fetching available files list')

        # Filter out already processed files
        files_to_extract = self.filter_existing_files(available_files)
        logging.debug('Finished filtering already extracted files')

        # Make destination dirs
        self.make_dirs(files_to_extract)

        if not files_to_extract:
            logging.debug('No files to extract')
            return

        # Extract each file
        self.extractor.start()
        for f in files_to_extract:
            self.extractor.extract(f, self.src_to_dest_path(f))

        self.extractor.join()


    def get_available_files(self):
        """
        Get list of all available files (within year_range) in src_dir.
        """
        available_files = []

        for year in range(self.year_range[0], self.year_range[1]+1):
            for month in range(1, 13):
                year_month = year_month_dir_fmt % (year, month)

                months_in_dir = [d for d in os.listdir(self.src_dir) if os.path.isdir(os.path.join(self.src_dir, d))]

                if year_month not in months_in_dir:
                    logging.debug('Missing Month: year %d month %d not in source' % (year, month))
                    continue

                days_in_month_dir = [d for d in os.listdir(os.path.join(self.src_dir, year_month)) if os.path.isdir(os.path.join(self.src_dir, year_month, d))]

                for day in range(1, du.days_per_month(month, du.is_leap_year(year))+1):
                    year_month_day = year_month_day_dir_fmt % (year, month, day)

                    if year_month_day not in days_in_month_dir:
                        logging.debug('Missing Day: year %d month %d day %d not in source' % (year, month, day))
                        continue

                    grib_dir_list = [d for d in os.listdir(os.path.join(self.src_dir, year_month, year_month_day)) if os.path.isfile(os.path.join(self.src_dir, year_month, year_month_day, d))]

                    todays_grib_files = [self.grib_file_fmt % (year_month_day, t, offset) for (t, offset) in time_offset_list]
                    for grib_file in todays_grib_files:
                        # Check if grib file not on server
                        if grib_file not in grib_dir_list:
                            logging.debug('Missing Grib: grib %s not in source' % grib_file)
                            continue

                        path = os.path.join(self.src_dir, year_month, year_month_day, grib_file)
                        available_files.append(path)

        return available_files

    def filter_existing_files(self, files):
        filtered_files = []

        for f in files:
            local_f = self.src_to_dest_path(f)
            if not os.path.isfile(local_f):
                filtered_files.append(f)

        return filtered_files

    def make_dirs(self, files):
        dirs = set(map(lambda x: os.path.dirname(self.src_to_dest_path(x)), files))
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


@click.command()
@click.argument('src_dir', type=click.Path(exists=True))
@click.argument('dest_dir', type=click.Path(exists=True))
@click.option('--start', default=2007, type=click.INT)
@click.option('--end', default=2016, type=click.INT)
@click.option('--scale', default='4', type=click.Choice([SCALE_HALF_DEG, SCALE_ONE_DEG]))
@click.option('--log', default='INFO')
def main(src_dir, dest_dir, start, end, scale, log):
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info('Reading data from %s. Storing data in "%s". Range is [%d, %d].' % (src_dir, dest_dir, start, end))

    logging.info('Starting GFS extraction')
    GfsExtract(src_dir, dest_dir, start, end, scale).extract()
    logging.info('End GFS extraction')

if __name__=='__main__':
    main()
