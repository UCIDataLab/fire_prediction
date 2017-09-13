"""
Base class for predictive models.
"""

class Model(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
