import numpy as np
import scipy.sparse as sp


def cluster_fires(df, thresh):
    """ Cluster a set of fire detections into a set of discrete fire events
    :param df: a DataFrame with x and y fields
    :param thresh: a threshold below which we consider two fire detections to be part of the same fire
    :return: n_fires, fires: number of fires detected and a vector of cluster assignments
    """
    xy_mat = np.transpose(np.array((np.array(df.y),np.array(df.x))))
    N = len(df)
    thresh_graph = sp.lil_matrix((N,N))
    for i in xrange(N):
        p_i = np.array([df.iloc[i].y, df.iloc[i].x])
        dist_arr = np.linalg.norm(xy_mat - p_i, axis=1)
        thresh_arr = dist_arr < thresh
        thresh_graph[i,:] = thresh_arr
    n_fires, fires = sp.csgraph.connected_components(thresh_graph, directed=False)
    return n_fires, fires
