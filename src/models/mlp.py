import sklearn.neural_network as sknn

from .base.model import Model


class MultilayerPerceptron(Model):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularizer_weight=None, normalize_params=False, t_k=None, add_exposure=False):
        super().__init__()
        self.response_var = response_var
        self.covariates = covariates
        self.log_covariates = log_covariates
        self.inputs = self.covariates + self.log_covariates
        self.variables = [self.response_var] + self.inputs
        self.fit_result = sknn.MLPRegressor(hidden_layer_sizes=(32,))

    def fit(self, X, y=None):
        y = X[self.response_var]
        X = X[self.inputs]
        self.fit_result.fit(X, y)

        return self

    def predict(self, X, choice=None):
        X = X[self.inputs]
        return self.fit_result.predict(X)
