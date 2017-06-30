import random
import cPickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import KDTree
FIRE_SEASON = (133,242)


def cluster_fires(df, thresh, return_df=False):
    """ Cluster a set of fire detections into a set of discrete fire events
    :param df: a DataFrame with x and y fields
    :param thresh: a threshold below which we consider two fire detections to be part of the same fire
    :return: n_fires, fires: number of fires detected and a vector of cluster assignments
    """
    min_year = int(np.min(df.year))
    max_year = int(np.max(df.year))
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


def find_daily_clusters(daily_df, kd_dict, clust2nodes=None, nodes2clust=None, merge_dict=None):
    """ helper function for cluster_over_time_with_merging to find all your boiz
    :param daily_df: pandas DataFrame with fires from today
    :param kd_dict: a dictionary based on the results of KDTree.query_ball_tree()
    :param preexisting_clusters: a map from fire index to cluster ID if we already have one
    :return: clust2nodes, nodes2clust, merge_dict
    """
    if len(daily_df):
        day = daily_df.iloc[0].dayofyear
    if clust2nodes is None:
        clust2nodes = dict()
        nodes2clust = dict()
        merge_dict = dict()
    new_clusters = set()

    for fire_id in daily_df.index:
        my_potential_clusters = set()
        if fire_id in kd_dict:
            neighbors = kd_dict[fire_id]
            for neighbor in neighbors:
                if neighbor in nodes2clust:
                    my_potential_clusters.add(nodes2clust[neighbor])
        if len(my_potential_clusters) == 0:
            if len(clust2nodes):
                new_clust_name = max(merge_dict.keys() + clust2nodes.keys()) + 1
            else:
                new_clust_name = 0
            nodes2clust[fire_id] = new_clust_name
            clust2nodes[new_clust_name] = set()
            clust2nodes[new_clust_name].add(fire_id)
            new_clusters.add(new_clust_name)
        elif len(my_potential_clusters) == 1:
            my_clust = list(my_potential_clusters)[0]
            nodes2clust[fire_id] = my_clust
            clust2nodes[my_clust].add(fire_id)
        else:
            # First, check if all but one of the clusters are new. Then, no need to merge!
            preexisting_clusts = my_potential_clusters.difference(new_clusters)
            if len(preexisting_clusts) == 1:
                # keep the old-timer!
                clust_to_keep = list(preexisting_clusts)[0]
            else:
                # pick the biggest
                inds = list(my_potential_clusters)
                clust_to_keep = inds[np.argmax(map(lambda x: len(clust2nodes[x]), inds))]
            for clust in my_potential_clusters.difference({clust_to_keep}):
                clust2nodes[clust_to_keep] = clust2nodes[clust_to_keep].union(clust2nodes[clust])
                for node in clust2nodes[clust]:
                    nodes2clust[node] = clust_to_keep
                del clust2nodes[clust]
                if clust not in new_clusters:
                    merge_dict[clust] = (clust_to_keep, day)
            nodes2clust[fire_id] = clust_to_keep
            clust2nodes[clust_to_keep].add(fire_id)
    return clust2nodes, nodes2clust, merge_dict


def cluster_over_time_with_merging(df, thresh, outfi=None):
    """ Cluster a set of fire detections over time
    :param df: DataFrame active fire detections with x and y variables
    :param thresh: threshold for clustering
    :return: (df, merge_dict_dict): the DataFrame with a cluster id added on and a dictionary of merged clusters
    """
    min_year = int(np.min(df.year))
    max_year = int(np.max(df.year))
    clust_dict = dict()
    merge_dict_dict = dict()

    for year in xrange(min_year, max_year+1):
        # Build up dictionary of nearest neighbors to make the building process easier
        annual_fires = df[df.year == year]
        annual_kd = KDTree(np.column_stack((annual_fires.x, annual_fires.y)))
        pairs_list = annual_kd.query_pairs(thresh)
        print "len of pairs list: " + str(len(pairs_list))
        neighbors_dict = dict()
        for i,j in pairs_list:
            i_name, j_name = (annual_fires.index[i], annual_fires.index[j])
            if i_name not in neighbors_dict:
                neighbors_dict[i_name] = set()
            neighbors_dict[i_name].add(j_name)
            if j_name not in neighbors_dict:
                neighbors_dict[j_name] = set()
            neighbors_dict[j_name].add(i_name)
        print "done building dict"
        # Get initial clusters on day 1
        first_day = max(FIRE_SEASON[0], np.min(annual_fires.dayofyear))
        last_day = min(FIRE_SEASON[1], np.max(annual_fires.dayofyear))
        clust2nodes, nodes2clust, merge_dict = (None, None, None)
        for day in xrange(first_day, last_day):
            daily_fires = annual_fires[annual_fires.dayofyear == day]
            clust2nodes, nodes2clust, merge_dict = find_daily_clusters(daily_fires, neighbors_dict,
                                                                       clust2nodes=clust2nodes, nodes2clust=nodes2clust,
                                                                       merge_dict=merge_dict)
            for node,clust in nodes2clust.iteritems():
                if node not in clust_dict:
                    clust_dict[node] = clust

        merge_dict_dict[year] = merge_dict
        print "%d merges in year %d" % (len(merge_dict), year)
    df['cluster'] = pd.Series(clust_dict, dtype=int)

    if outfi:
        with open(outfi) as fout:
            cPickle.dump(df, fout, cPickle.HIGHEST_PROTOCOL)
    return df, merge_dict_dict
