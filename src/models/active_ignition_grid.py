"""
Model for predicting detections in a gridded region by combining an active fire and ignition model.
"""

import numpy as np

from .base.model import Model


class ActiveIgnitionGridModel(Model):
    def __init__(self, active_fire_model, ignition_model):
        super(ActiveIgnitionGridModel, self).__init__()

        self.afm = active_fire_model
        self.igm = ignition_model

    def fit(self, X, y=None):
        fit_model = [None, None]
        if self.afm:
            fit_model[0] = self.afm.fit(X[0])
        if self.igm:
            fit_model[1] = self.igm.fit(X[1])

        # return tuple(fit_model)
        return self

    def predict(self, X, shape=None):
        pred = np.zeros(shape)

        if self.afm:
            pred += self.afm.predict(X[0], shape)

        if self.igm:
            pred += self.igm.predict(X[1], shape)

        return pred
