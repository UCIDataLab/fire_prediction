"""
Helper functions for dealing with calendar dates.
"""
from datetime import timedelta, datetime

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
    Generate all dates between start and end date.

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

