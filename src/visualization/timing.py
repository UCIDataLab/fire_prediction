"""
Generating timing related visualizations.
"""

from matplotlib import pyplot as plt

from src.helper import date_util as du


def plot_df(df, data_types, title=''):
    fig, axes = plt.subplots(nrows=len(data_types), ncols=1, figsize=(12, 10))
    plt.suptitle(title)
    plt.tight_layout(pad=4)

    for i, (type_, form, title) in enumerate(data_types):
        axes[i].plot(map(lambda x: du.day_of_year_from_datetime(x), df.date_local), df[type_], form)
        axes[i].set_title(title)
