import numpy as np
import scipy.sparse as sp


def cluster_fires(df, thresh, return_df=False):
    """ Cluster a set of fire detections into a set of discrete fire events
    :param df: a DataFrame with x and y fields
    :param thresh: a threshold below which we consider two fire detections to be part of the same fire
    :return: n_fires, fires: number of fires detected and a vector of cluster assignments
    """
    min_year = np.min(df.year)
    max_year = np.max(df.year)
    df['cluster'] = 0
    n_fires = 0
    for year in xrange(min_year, max_year+1):
        year_df = df[df.year==year]
        xy_mat = np.transpose(np.array((np.array(year_df.y),np.array(year_df.x))))
        N = len(df)
        thresh_graph = sp.lil_matrix((N,N))
        for i in xrange(N):
            p_i = np.array([year_df.iloc[i].y, year_df.iloc[i].x])
            dist_arr = np.linalg.norm(xy_mat - p_i, axis=1)
            thresh_arr = dist_arr < thresh
            thresh_graph[i,:] = thresh_arr
        n_fires_annual, fires = sp.csgraph.connected_components(thresh_graph, directed=False)
        big_fires = fires + n_fires
        df.loc[year_df.index, 'cluster'] = big_fires
        n_fires += n_fires_annual

    if return_df:
        return df
    return n_fires, df.cluster
