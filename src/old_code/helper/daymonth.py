import datetime

def utc_to_local_time(datetime_utc, longitude):
    """ 
    Calculate local time based on longitude and utc time. Rounded to nearest second.
    """
    timedelta_offset = datetime.timedelta(0, round(longitude * 4 * 60))
    return datetime_utc + timedelta_offset

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
