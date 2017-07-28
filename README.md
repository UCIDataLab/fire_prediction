# fire_prediction
Using MODIS active fire data and NOAA GFS data to predict the spread of forest fires, particularly in Alaska.

The project is organized as such:
* carpentry: converting raw data into more usable formats
  * get_modis_data: Convert MODIS active fire dataset from local TSVs to a pandas DataFrame (tutorial in tutorial folder). A copy of the raw data exists on /extra/zbutler0/data/mcd14ml.tar.gz
  * get_gfs_data: Pull GFS data from NOAA servers and save only the relevant layers for relevant locations into a dictionary (tutorial in tutorial folder). Some, but not all, of the raw data exists in /extra/zbutler0/data/gfs/
  * make_cluster_df: Make a cluster feature DataFrame from a MODIS DataFrame and GFS dict
  * read_station_data: Read data from a weather station CSV from NOAA NCDC. An example raw CSV exists at /extra/zbutler0/data/fairbanks.csv, and its corresponding DataFrame at /extra/zbutler0/data/fairbanks.pkl. Note that there's some hackery in here: in this particular file, we found that accurate precipitation data existed on the last timestamp of a day but not elsewhere--this might not hold true elsewhere!
  * get_burn_data: Get MODIS burned area product. I haven't done anything with this data yet--all this function does is pull the data from the server.
* data: Includes wrapper functions for loading data. You'll need to add your own paths though
* exploratory: Exploratory iPython notebooks. The older ones use older data standards and are thus not necessarily reproducible.
* geometry: Contains fire_clustering which adds a "cluster" column to a MODIS active fire DataFrame and grid_conversion, which contains several helper functions for dealing with an XY grid instead of lat/lon
* plotting: functions/iPython notebooks to produce various plots.
* prediction: contains model code. Currently our only style of model is ClusterRegression
* testing: Unit tests
* tutorial: iPython notebooks to acquaint user with this package
* util: various utilites
