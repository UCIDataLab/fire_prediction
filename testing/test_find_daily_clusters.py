import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from geometry.fire_clustering import find_daily_clusters

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from pylab import get_current_fig_manager


def test_find_daily_clusters(plot=True):
    df_dict = dict()
    df_dict['x'] = [1, 2, 2, 3, 10, 10, 10, 10, 10, 10, .5, .8, 2.5, 2.7, 10, 11, 12, 5, 5.5, 5, 5.5]
    df_dict['y'] = [1, .5, 1.5, 1.2, 1, 2, 3, 5, 6, 7, 2, 1.2, .5, .6, 4, 4, 4, 3, 3, 3.3, 3.3]
    df_dict['dayofyear'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    df = pd.DataFrame(df_dict)
    print "DataFrame:"
    print df
    if plot:
        plt.scatter(df.x, df.y, c=df.dayofyear)
        get_current_fig_manager().window.raise_()
        plt.show()
        plt.close()

    kd = KDTree(np.column_stack((df.x, df.y)))
    neighbors_list = kd.query_ball_tree(kd, 1.05)
    neighbors_dict = dict()
    for i,arr in enumerate(neighbors_list):
        neighbors_dict[df.iloc[i].name] = set()
        for j in arr:
            neighbors_dict[df.iloc[i].name].add(df.iloc[j].name)

    # Get initial clusters on day 1
    day1fires = df[df.dayofyear == 1]
    clust2nodes, nodes2clust, merge_dict = find_daily_clusters(day1fires, neighbors_dict)
    print "clust2nodes, day 1:" + str(clust2nodes)
    print "node2clusts, day 1:" + str(nodes2clust)
    print "merge_dict, day 1:" + str(merge_dict)

    day2fires = df[df.dayofyear == 2]
    clust2nodes, node2clusts, merge_dict = find_daily_clusters(day2fires, neighbors_dict, clust2nodes=clust2nodes,
                                                               nodes2clust=nodes2clust, merge_dict=merge_dict)
    print "clust2nodes, day 2:" + str(clust2nodes)
    print "node2clusts, day 2:" + str(nodes2clust)
    print "merge_dict, day 2:" + str(merge_dict)

    day3fires = df[df.dayofyear == 3]
    clust2nodes, node2clusts, merge_dict = find_daily_clusters(day3fires, neighbors_dict, clust2nodes=clust2nodes,
                                                               nodes2clust=nodes2clust, merge_dict=merge_dict)
    print "clust2nodes, day 3:" + str(clust2nodes)
    print "node2clusts, day 3:" + str(nodes2clust)
    print "merge_dict, day 3:" + str(merge_dict)


if __name__ == "__main__":
    test_find_daily_clusters()