import matplotlib.pyplot as plt

title_dict = dict()  # Titles for subplots of each potential variable
title_dict['n_det'] = "Number of detections"
title_dict['temp'] = "Noon temperature"
title_dict['humidity'] = "Noon relative humidity"
title_dict['vpd'] = "Noon Vapor Pressure Defecit (VPD)"
title_dict['wind'] = "Noon wind speed"
title_dict['rain'] = "Daily precipitation"


def covar_plots(year, date_range=(133,242), modis_df=None, gfs_df=None, station_df=None,
                gfs_marker='b-', station_marker='r-', outfi=None,
                covars=('n_det', 'temp', 'humidity', 'vpd', 'wind', 'rain')):
    """ Make plot of covariates, optionally across satellite and station data
    :param year: Year we want to plot
    :param date_range: Range of dates in that year we want to plot (default to Alaska fire season)
    :param modis_df: pandas DataFrame with GFS data
    :param station_df: pandas DataFrame with station data
    :param gfs_marker: marker for GFS covariates
    :param station_marker: marker for station covariates
    :param outfi: file to write plots to
    :param covars: list of covariates to use, default to all
    :return:
    """
    if modis_df is not None:
        modis_season = modis_df[(modis_df.year == year) & (modis_df.dayofyear > date_range[0]) &
                                (modis_df.dayofyear < date_range[1])]
    if gfs_df is not None:
        gfs_season = gfs_df[(gfs_df.year == year) & (gfs_df.dayofyear > date_range[0]) &
                                (gfs_df.dayofyear < date_range[1])]
    if station_df is not None:
        station_season = station_df[(station_df.year == year) & (station_df.dayofyear > date_range[0]) &
                                    (station_df.dayofyear < date_range[1])]

    fig, axes = plt.subplots(nrows=len(covars), ncols=1, figsize=(12,10))
    for i, var in enumerate(covars):
        if not i:
            ax = plt.subplot(511 + i)
            ax1 = ax
        else:
            ax = plt.subplot(511 + i, sharex=ax1)

        if var == 'n_det':
            if modis_df is not None:
                plt.plot(modis_season.dayofyear, modis_season[var], gfs_marker)
        else:
            if gfs_df is not None:
                plt.plot(gfs_season.dayofyear, gfs_season[var], gfs_marker)
            if station_df is not None:
                plt.plot(station_season.dayofyear, station_season[var], station_marker)
        plt.title(title_dict[var])

    fig.tight_layout()
    plt.savefig(outfi)
