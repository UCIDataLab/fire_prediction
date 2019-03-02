"""
Model for predicting detections in a gridded region by combining multiple models.
"""

import numpy as np

from .base.model import Model


class MultiGroupGridModel(Model):
    def __init__(self, col_name, val_model_dict):
        super(MultiGroupGridModel, self).__init__()

        self.col_name = col_name
        self.val_model_dict = val_model_dict

    def fit(self, X, y=None):
        fit_model = {}
        for val, model in self.val_model_dict.items():
            X_group = self.filter_data(X, val)
            fit_model[val] = model.fit(X_group)

        return fit_model.items()

    def predict(self, X, shape=None):
        pred = np.zeros(shape)

        for val, model in self.val_model_dict.items():
            X_group = self.filter_data(X, val)
            pred += model.predict(X_group, shape)

        return pred

    @staticmethod
    def filter_data(X, val):
        return X
