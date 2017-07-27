from prediction.cluster_regression import add_autoreg_and_n_det
from data import data


def test_add_autoreg_and_t_k():
    clust_feat_df = data.load_clust_feat_df(clust_thresh=5)
    clust_feat_df = add_autoreg_and_n_det(clust_feat_df, 3, 3)
