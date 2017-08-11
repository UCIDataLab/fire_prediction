# Test to make sure clusters under both systems are similar
for clust_id in njc.cluster.unique():
    if np.isnan(clust_id):
        continue
    this_clust_df = njc[njc.cluster == clust_id]
    examp = this_clust_df.index[0]
    clust_in_old = just_clust_df.loc[examp].cluster
    clusts_in_old = just_clust_df.loc[this_clust_df.index].cluster.unique()
    if len(clusts_in_old) != 1:
        print "mismatch: clust %d in new matches with clusts %s in old" % (clust_id, str(clusts_in_old))
