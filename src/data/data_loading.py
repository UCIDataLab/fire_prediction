import cPickle
from os.path import join


def load_modis(data_dir="data/"):
    with open(join(data_dir, "ak_fires.pkl")) as fpkl:
        return cPickle.load(fpkl)


def load_fb_station(data_dir="data/"):
    with open(join(data_dir, "stations", "fairbanks.pkl")) as fpkl:
        return cPickle.load(fpkl)


def load_gfs_dict(data_dir="data/"):
    with open(join(data_dir, "weather", "gfs_ak_dict.pkl")) as fpkl:
        return cPickle.load(fpkl)


def load_fb_gfs(data_dir="data/"):
    with open(join(data_dir, "weather", "fb_gfs.pkl")) as fpkl:
        return cPickle.load(fpkl)


def load_clust_df(data_dir="data/", clust_thresh=10):
    with open(join(data_dir, "clusters", "clust_df_%d.pkl" % clust_thresh)) as fpkl:
        return cPickle.load(fpkl)


def load_merge_dict(data_dir="data/", clust_thresh=10):
    with open(join(data_dir, "clusters", "merge_dict_%d.pkl" % clust_thresh)) as fpkl:
        return cPickle.load(fpkl)


def load_clust_feat_df(data_dir="data/", clust_thresh=10):
    with open(join(data_dir, "clusters", "clust_feat_df_%d.pkl" % clust_thresh)) as fpkl:
        return cPickle.load(fpkl)
