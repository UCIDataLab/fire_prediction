"""
Helper functions for dealing with calendar dates.
"""
import math
from datetime import timedelta, datetime, tzinfo
import pytz
from functools import total_ordering

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
    return month_arr[month-1]

def daterange(start_date, end_date=None, increment=timedelta(hours=24)):
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

def round_to_nearest_multiple(x, multiple):
    return multiple * round(x/multiple)

def round_to_nearest_quarter_hour(x):
    return round_to_nearest_multiple(x, .25)

def utc_to_local_time_offset(lon, round_func=round_to_nearest_quarter_hour):
    """
    Returns utc to local offset time in hours.

    By default rounds to nearest quarter hour.
    """
    offset = (lon * (12./math.pi)) / 60. 

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

def day2monthday(my_day, leapyear=False):
    if leapyear:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = 1
    days_left = my_day
    while days_left >= month_arr[month-1]:
        days_left -= month_arr[month-1]
        month += 1
    day = days_left + 1
    return month,day


def monthday2day(month, day, leapyear=False):
    """Convert month/day into days since Jan 1"""
    if leapyear:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = 0
    for mon in xrange(1, month):
        days += month_arr[mon - 1]
    days += day - 1
    return days


def increment_day(year, month, day):
    if year % 4:  # not leap year
        days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    else:  # leap year
        days_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month

    if day == days_arr[month-1]:
        day = 1
        month += 1
        if month == 13:
            month = 1
            year += 1
    else:
        day += 1

    return year, month, day


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
    Tracks a datetime with an offset (usually used for integrated measurements) and supports conversion to truly local time.

    Stores all datetimes in UTC local. Supports using lon for a truly localized offset
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

        WARNGING: Cannot pickle a truly localized datetime because it relies on a tzinfo wich requires init() arguments.
        """
        if lon:
            return self.dt.astimezone(self.get_localized_tz(lon))

        return self.dt

    def get_offset(self):
        return self.td_offset

    def get_localized_tz(self, lon):
        if self.lon is None:
            raise ValueError('lon must be defined to get a truly localized tzinfo')

        return TrulyLocalTzInfo(lon)

    def __eq__(self, other):
        return self.get() == other.get()

    def __lt__(self, other):
        return self.get() < other.get()


def day2monthday(my_day, leapyear=False):
    if leapyear:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = 1
    days_left = my_day
    while days_left >= month_arr[month-1]:
        days_left -= month_arr[month-1]
        month += 1
    day = days_left + 1
    return month,day


def monthday2day(month, day, leapyear=False):
    """Convert month/day into days since Jan 1"""
    if leapyear:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = 0
    for mon in xrange(1, month):
        days += month_arr[mon - 1]
    days += day - 1
    return days


def increment_day(year, month, day):
    if year % 4:  # not leap year
        days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    else:  # leap year
        days_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month

    if day == days_arr[month-1]:
        day = 1
        month += 1
        if month == 13:
            month = 1
            year += 1
    else:
        day += 1

    return year, month, day
