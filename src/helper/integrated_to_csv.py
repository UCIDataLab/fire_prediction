import csv
import sys
import pickle

import datetime
import StringIO

def date_to_day_of_year(date):
    year,month,day = map(int, date.split('-'))
    return year,datetime.date(year, month, day).timetuple().tm_yday

with open(sys.argv[1], 'rb') as fin:
    df = pickle.load(fin)
    df['num_det_target'] = df['num_det_target'].astype('int32')
    csv_file = StringIO.StringIO(df.to_csv())

out = ''

reader = csv.reader(csv_file, delimiter=',')
header = reader.next()
out += 'year,day_of_year,' + ','.join(header[2:4]) + ','+ ','.join(header[8:]) + '\n'

for row in reader:
    year,day_of_year = date_to_day_of_year(row[1])
    line = '%d,%d,%s,%s\n' % (year,day_of_year,','.join(row[2:4]), ','.join(row[8:]))

    out += line


with open(sys.argv[2], 'wb') as fout:
    fout.write(out)
