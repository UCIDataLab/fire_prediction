"""
Models designed to wrap simpler models to allow predictions onto a grid.
"""

from io import StringIO

import numpy as np
import pandas as pd

from .base.model import Model

large_diffs = []
large_diffs2 = []


# count = 0

def convert_pred_to_grid(pred, shape):
    pred = np.array(pred)
    pred = np.reshape(pred, shape, order='F')

    return pred


def prep_df_for_model(X, filter_func):
    if not isinstance(X, pd.DataFrame):
        X_df = X.to_dataframe()
    else:
        X_df = X

    if filter_func:
        X_df = filter_func(X_df)

    if 'filter_mask' in X_df.columns:
        X_df = filter_mask(X_df)

    X_df = pd.read_csv(StringIO(X_df.to_csv()))

    return X_df


class GridComponent(Model):
    def __init__(self, model, filter_func=None, pred_func=None):
        super().__init__()

        self.model = model
        self.filter_func = filter_func
        self.pred_func = pred_func

    def fit(self, X, y=None):
        X = prep_df_for_model(X, self.filter_func)
        self.model = self.model.fit(X)

        return self

    def predict(self, X, shape=None, choice=None):
        # If passing xarray, convert to dataframe
        try:
            X_df = X.to_dataframe()
        except AttributeError:
            X_df = X

        rem = None
        if choice is not None:
            ret = self.model.predict(X_df, choice)
            pred = ret[0]
            rem = ret[1:]
            print('g', np.shape(rem))
        else:
            pred = self.model.predict(X_df, choice)

        if self.pred_func:
            pred = self.pred_func(X_df, pred)

        if shape is not None:
            pred = convert_pred_to_grid(pred, shape)

        if choice is not None:
            return pred, rem
        else:
            return pred

    def get_model(self):
        return self.model.get_model()


class UnifiedGrid(GridComponent):
    def fit(self, X, y=None):
        return super().fit(X[0], y)

    def predict(self, X, shape=None, choice=None):
        return super().predict(X[0], shape)


def filter_mask(x):
    return x[x.filter_mask]


def active_filter(x):
    return x[x.active]


def ignition_filter(x):
    return x[~x.active]


def active_pred(x, y):
    return y * x.active


def ignition_pred(x, y):
    return y * (~x.active)


class ActiveIgnitionGrid(Model):
    def __init__(self, active_fire_model, ignition_model, active_filter_func=active_filter,
                 ignition_filter_func=ignition_filter, active_pred_func=active_pred,
                 ignition_pred_func=ignition_pred):
        super().__init__()

        afm_filter_func = active_filter_func
        igm_filter_func = ignition_filter_func

        afm_pred_func = active_pred_func
        igm_pred_func = ignition_pred_func

        if active_fire_model:
            self.afm = GridComponent(active_fire_model, afm_filter_func, afm_pred_func)
        else:
            self.afm = None

        if ignition_model:
            self.igm = GridComponent(ignition_model, igm_filter_func, igm_pred_func)
        else:
            self.igm = None

    def fit(self, X, y=None):
        # Train active fire component
        if self.afm:
            self.afm = self.afm.fit(X[0], y)

        # Train ignition component
        if self.igm:
            self.igm = self.igm.fit(X[1], y)

        return self

    def predict(self, X, shape=None, choice=None):
        rem = []

        if shape is None:
            pred = np.zeros(len(X[0]))
        else:
            pred = np.zeros(shape)

        # Predict with active fire component
        if self.afm:
            if choice is not None:
                ret_afm = self.afm.predict(X[0], shape, choice)

                pred_afm = ret_afm[0]
                pred += pred_afm

                # rem_afm = ret_afm[1:]
                print('ai', np.shape(ret_afm[1]))
                rem.append(ret_afm[1])
            else:
                pred += self.afm.predict(X[0], shape, choice)

        # Predict with ignition component
        if self.igm:
            if choice is not None:
                ret_igm = self.igm.predict(X[1], shape, choice)

                pred_igm = ret_igm[0]
                pred += pred_igm

                # rem_igm = ret_igm[1:]
                rem.append(ret_igm[1])
            else:
                pred += self.igm.predict(X[1], shape, choice)

        if choice is not None:
            return pred, rem
        else:
            return pred

    def get_model(self):
        models = {}
        if self.afm:
            models['active'] = self.afm.get_model()
        if self.igm:
            models['ignition'] = self.igm.get_model()

        return models


class SwitchingRegressionGrid(Model):
    def __init__(self, active_fire_model, ignition_model, mixture_model, active_filter_func=active_filter,
                 ignition_filter_func=ignition_filter):
        super().__init__()

        afm_filter_func = active_filter_func
        igm_filter_func = ignition_filter_func

        if active_fire_model:
            self.afm = GridComponent(active_fire_model, afm_filter_func, None)
        else:
            self.afm = None

        if ignition_model:
            self.igm = GridComponent(ignition_model, igm_filter_func, None)
        else:
            self.igm = None

        self.mixture_model = GridComponent(mixture_model, None, None)

    def fit(self, X, y=None):
        # Train active fire component
        if self.afm:
            self.afm = self.afm.fit(X[0], y)

        # Train ignition component
        if self.igm:
            self.igm = self.igm.fit(X[1], y)

        print('z', X[2].large_fire.shape)
        self.mixture_model.fit(X[2], y)

        return self

    def predict(self, X, shape=None, choice=None):
        rem = []

        if shape is None:
            pred = np.zeros(len(X[0]))
        else:
            pred = np.zeros(shape)

        mixture_pred = self.mixture_model.predict(X[2], shape, choice)

        large = X[2].large_fire.values[X[2].active]
        large_pred = (mixture_pred.values > .5).astype(np.int)[X[2].active]
        large_pred2 = mixture_pred.values[X[2].active]
        print(large.shape, large_pred.shape)
        large_diff = np.mean(np.absolute(large - large_pred))
        large_diff2 = np.mean(np.absolute(large - large_pred2))
        large_diffs.append(large_diff)
        large_diffs2.append(large_diff2)
        print('dif', large_diffs)
        print('dif2', large_diffs2)
        print('mix mean', np.mean(large_pred))
        print('mix mean 2', np.mean(mixture_pred.values[X[2].active]))
        print('large mean', np.mean(large))
        large2 = X[2].large_fire.values[X[2].num_det > 0]
        print('large mean', np.mean(large2))

        """
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        h = plt.hist(mixture_pred.values[X[2].active], bins=100)
        global count
        plt.savefig('hist_%d.png' % count)
        plt.gcf().clear()
        plt.title('Pred. Probability (Given Large Fire Observed, 90% split)')
        h = plt.hist(mixture_pred.values[X[2].active.values & X[2].large_fire.values], bins=100)
        print('s0',mixture_pred.values[X[2].active.values & X[2].large_fire.values].shape)
        plt.savefig('hist_large_%d.png' % count)
        plt.gcf().clear()
        plt.title('Pred. Probability (Given Small Fire Observed, 90% split)')
        h = plt.hist(mixture_pred.values[X[2].active.values & ~X[2].large_fire.values], bins=100)
        print('s1',mixture_pred.values[X[2].active.values & ~X[2].large_fire.values].shape)
        plt.savefig('hist_small_%d.png' % count)
        plt.gcf().clear()
        count += 1
        """

        # Toggle if prob of large > .5
        # mixture_pred = (mixture_pred.values>.5).astype(np.int)

        # Predict with active fire component
        if self.afm:
            if choice is not None:
                ret_afm = self.afm.predict(X[0], shape, choice)

                pred_afm = ret_afm[0]
                pred += mixture_pred * pred_afm

                # rem_afm = ret_afm[1:]
                print('ai', np.shape(ret_afm[1]))
                rem.append(ret_afm[1])
            else:
                pred_afm = self.afm.predict(X[0], shape, choice)
                print('ss', pred_afm.shape)

                pred += mixture_pred * pred_afm

        # Predict with ignition component
        if self.igm:
            if choice is not None:
                ret_igm = self.igm.predict(X[1], shape, choice)

                pred_igm = ret_igm[0]
                pred += (1 - mixture_pred) * pred_igm

                # rem_igm = ret_igm[1:]
                rem.append(ret_igm[1])
            else:
                pred_igm = self.igm.predict(X[1], shape, choice)
                print('sss', pred_igm.shape)

                pred += (1 - mixture_pred) * pred_igm

        rem.append(mixture_pred)

        if choice is not None:
            return pred, rem
        else:
            return pred

    def get_model(self):
        models = {}
        if self.afm:
            models['active'] = self.afm.get_model()
        if self.igm:
            models['ignition'] = self.igm.get_model()

        return models
