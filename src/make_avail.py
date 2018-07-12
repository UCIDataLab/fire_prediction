import pickle
import sys
from collections import defaultdict
import os
import parse


def get_year(fn):
    fmt = 'gfsanl_{resolution}_{date}_{time:d}_{offset:d}.{}' 
    p = parse.parse(fmt, fn)

    year,month,day = int(p['date'][:4]), int(p['date'][4:6]), int(p['date'][6:8])

    return year

with open(sys.argv[1], 'rb') as fin:
    avail = pickle.load(fin)

years = defaultdict(list)
for a in avail:
    year = get_year(os.path.split(a)[1])
    years[year].append(a)


for k,v in years.items():
    dirname, _ = os.path.split(sys.argv[1])

    fn = 'gfsanl_3_available_0101%d-1231%d.pkl' % (k, k)
    with open(os.path.join(dirname, fn), 'wb') as fout:
        pickle.dump(v, fout)
