"""
Model wrappers for making advanced forecast predictions (e.g. recursive forecasting).
"""

from src.helper import multidata_wrapper as mdw
from .base.model import Model


class RecursiveForecast(Model):
    def __init__(self, model, t_k):
        super().__init__()
        self.model = model
        self.t_k = t_k

    def fit(self, X, y=None):
        """ Fit on the t_k = 1 data. """
        X = mdw.MultidataWrapper((X[0], X[0]))
        self.model = self.model.fit(X, y)

        return self

    def predict(self, X, shape=None):
        """ Predict by recursively predicting using the previous predictions as input. """
        # TODO: Support update memory according to predictions
        pred = None
        for i in range(1, self.t_k + 1):
            if i > 1:
                X_ds = X[i - 1].assign(num_det=(('y', 'x', 'time'), pred))
            else:
                X_ds = X[0]

            X_cur = mdw.MultidataWrapper((X_ds, X_ds))
            pred = self.model.predict(X_cur, shape)

        return pred
