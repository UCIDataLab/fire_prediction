from prediction.fire_clustering import cluster_over_time_with_merging


class ClusterRegression:
    """ Class for performing cluster regression for fire prediction
    """
    def __init__(self, modis_df, gfs_dict, clust_thresh, grid_size):
        """ Set up cluster model--build clusters and grid cells (if desired)
        :param modis_df: DataFrame with MODIS active fire detections
        :param gfs_dict: Dictionary with tensors of GFS weather variables
        :param clust_thresh: threshold (in KM) of where two detections are considered the same fire
        :param grid_size: size of grid on which to predict new nodes. If grid_size <= 0, don't have a grid component
        :return:
        """
        clust_df = cluster_over_time_with_merging(modis_df, clust_thresh)
        self.clust_df = add_gfs(clust_df, gfs_dict)
        self.clust_thresh = clust_thresh
        self.grid_size = grid_size

    def fit(self, train_years):
        """ Train on the specified years
        :param train_years: Iterable containing list of years to train on
        :return:
        """

    def predict(self, test_years):
        """ Get predictions for specified
        :param test_years: Iterable containing list of years to test on
        :return:
        """
