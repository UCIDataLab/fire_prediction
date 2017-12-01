"""
Model for predicting detections in a gridded region by combining an active fire and ignition model.
"""

import numpy as np

from base.model import Model

class ActiveIgnitionGridModel(Model):
    def __init__(self, active_fire_model, ignition_model):
        super(ActiveIgnitionGridModel, self).__init__()

        self.afm = active_fire_model
        self.igm = ignition_model

    def fit(self, X, y=None):
        if self.afm:
            self.afm.fit(X[0])
        if self.igm:
            self.igm.fit(X[1])

    def predict(self, X):
        pred = np.zeros(X[1].shape[:3])

        if self.afm:
            pred += self.afm.predict(X[0])

        if self.igm:
            pred += self.igm.predict(X[1])

        return pred
