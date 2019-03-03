"""
Helper functions for dealing with calendar dates.
"""
import datetime as dtime
from datetime import timedelta, tzinfo
from functools import total_ordering

import numpy as np
import pandas as pd
import pytz
import xarray as xr

INC_ONE_DAY = timedelta(hours=24)


def is_leap_year(year):
    """
    Is the given year a leap year.
    """
    return year % 4 == 0


def days_per_month(month, is_leap):
    """
    Get number of days in given month. 

    Accounts for extra day in feb during leap year.
    """
    if is_leap:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return month_arr[month - 1]


def date_range(start_date, end_date=None, increment=timedelta(hours=24)):
    """
    Generate all dates between start and end date (inclusive lower, exclusive upper).

    Can use date or datetime.
    """
    if not end_date:
        while True:
            yield start_date
            start_date += increment
    else:
        while start_date < end_date:
            yield start_date
            start_date += increment


def date_range_days(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + dtime.timedelta(n)


def date_range_months(start_date, end_date):
    """Iterate through months in date range (inclusive)."""

    cur_month, cur_year = start_date.month, start_date.year
    end_month, end_year = end_date.month, end_date.year

    while (cur_month != end_month) or (cur_year != end_year):
        yield dtime.date(cur_year, cur_month, 1)

        cur_year += cur_month == 12
        cur_month = np.remainder(cur_month, 12) + 1

    yield dtime.date(end_year, end_month, 1)


def filter_fire_season(ds, start=(5, 14), end=(8, 31), years=range(2007, 2016 + 1)):
    dates = ds.time.values
    ind = np.zeros(dates.shape, dtype=np.bool)
    for year in years:
        start_date = np.datetime64(dtime.date(year, start[0], start[1]))
        end_date = np.datetime64(dtime.date(year, end[0], end[1]))
        ind = ind | ((dates >= start_date) & (dates <= end_date))

    data_vars = {}
    for k in ds.data_vars.keys():
        data_vars[k] = (('y', 'x', 'time'), np.array(ds[k].values)[:, :, ind], ds[k].attrs)

    new_ds = xr.Dataset(data_vars, coords={'time': ds.time.values[ind]}, attrs=ds.attrs)

    return new_ds


def create_true_dates(start_date, end_date, times, offsets):
    """ 
    Used to build list of datetime and offsets for a range of dates. Returns datetime64 dates.

    Inclusive of dates. 
    """
    dates = list(date_range_days(start_date, end_date + dtime.timedelta(1)))
    times = list(map(lambda x: dtime.time(x), times))
    offsets = list(map(lambda x: dtime.timedelta(hours=x), offsets))

    true_dates, true_offsets = zip(*[(dtime.datetime.combine(d, t), o) for d in dates for t in times for o in offsets])

    true_dates = pd.to_datetime(true_dates)
    return true_dates, np.array(true_offsets)


def round_to_nearest_multiple(x, multiple):
    return multiple * round(x / multiple)


def round_to_nearest_quarter_hour(x):
    return round_to_nearest_multiple(x, .25)


def utc_to_local_time_offset(lon, round_func=round_to_nearest_quarter_hour):
    """
    Returns utc to local offset time in hours.

    By default rounds to nearest quarter hour.
    """
    offset = (lon / 15.)

    if round_func:
        offset = round_func(offset)

    return timedelta(hours=offset)


def utc_to_local_time(datetime_utc, lon, round_func=None):
    """ 
    Calculate local time based on longitude and utc time.

    By default rounds to nearest quarter hour.
    """
    tz = TrulyLocalTzInfo(lon, round_func)

    return datetime_utc.astimezone(tz)


class TrulyLocalTzInfo(tzinfo):
    def __init__(self, lon, round_func=None):
        super(TrulyLocalTzInfo, self).__init__()
        self.lon = lon
        self.round_func = round_func

    def utcoffset(self, dt):
        offset = utc_to_local_time_offset(self.lon, self.round_func) + self.dst(dt)
        return offset

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return 'local'


@total_ordering
class DatetimeMeasurement(object):
    """
    Tracks a datetime with an offset (usually used for integrated measurements) and supports conversion to truly local
    time.

    Stores all datetime in UTC local. Supports using lon for a truly localized offset
    """

    def __init__(self, dt, td_offset=timedelta(0)):
        # Check that datetime dt is timezone aware
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError('Datetime object must be timezone aware')

        self.dt = dt.astimezone(pytz.UTC)
        self.td_offset = td_offset

    def get(self, lon=None):
        """
        Return datetime. Can be truly localized if lat/lon are provided.

        WARNING: Cannot pickle a truly localized datetime because it relies on a tzinfo which requires init() arguments.
        """
        if lon:
            return self.dt.astimezone(self.get_localized_tz(lon))

        return self.dt

    def get_offset(self):
        return self.td_offset

    @staticmethod
    def get_localized_tz(lon):
        if lon is None:
            raise ValueError('lon must be defined to get a truly localized tzinfo')

        return TrulyLocalTzInfo(lon)

    def __eq__(self, other):
        return self.get() == other.get()

    def __lt__(self, other):
        return self.get() < other.get()


def day2monthday(my_day, leap_year=False):
    if leap_year:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = 1
    days_left = my_day
    while days_left >= month_arr[month - 1]:
        days_left -= month_arr[month - 1]
        month += 1
    day = days_left + 1
    return month, day


def monthday2day(month, day, leap_year=False):
    """Convert month/day into days since Jan 1"""
    if leap_year:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = 0
    for mon in range(1, month):
        days += month_arr[mon - 1]
    days += day - 1
    return days


def increment_day(year, month, day):
    if year % 4:  # not leap year
        days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    else:  # leap year
        days_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month

    if day == days_arr[month - 1]:
        day = 1
        month += 1
        if month == 13:
            month = 1
            year += 1
    else:
        day += 1

    return year, month, day


def day_of_year_from_datetime(dt_):
    return dt_.timetuple().tm_yday
