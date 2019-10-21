import numpy as np

import sklearn.neural_network as sklearn_nn

from src.models.regression_models import BasicModelBase


class MultilayerPerceptron(BasicModelBase):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None, add_exposure=False, extra_model_params=None):
        super().__init__(response_var, covariates, log_covariates, log_correction, log_correction_const,
                         regularization_weight, normalize_params, t_k, add_exposure)
        self.inputs = self.covariates + self.log_covariates
        self.variables = [self.response_var] + self.inputs

        if extra_model_params:
            print('Extra Model Args', extra_model_params)
            hidden_layer_sizes = extra_model_params
        else:
            hidden_layer_sizes = (32,)

        self.fit_result = sklearn_nn.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)

    def fit(self, X, y=None):
        y = X[self.response_var]
        X = X[self.inputs]
        self.fit_result.fit(X, y)

        return self

    def predict(self, X, choice=None):
        X = X[self.inputs]
        return self.fit_result.predict(X)
