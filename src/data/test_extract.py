import sys
import cPickle as pickle
import grib
import logging


def extract_thread(context, src_path, dest_path):
    logging.debug('Extracting %s' % src_path)
    selector = context['selector']

    with open(src_path, 'rb') as fin:
        grib_file = grib.GribFile(fin)
        try:
            extracted = selector.select(grib_file)
        except Exception as err:
            logging.debug('Exception "%s" encountered while selecting from file "%s"' % (str(err), src_path))
            extracted = None

    if extracted:
        with open(dest_path, 'wb') as fout:
            pickle.dump(extracted, fout, protocol=pickle.HIGHEST_PROTOCOL)

log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=getattr(logging, 'DEBUG'), format=log_fmt)

selections = grib.get_default_selections()
bb = grib.get_default_bounding_box()
context = {}
context['selector'] = grib.GribSelector(selections, bb)
extract_thread(context, sys.argv[1], sys.argv[2])
