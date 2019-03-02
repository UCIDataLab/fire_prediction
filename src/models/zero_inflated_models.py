import numpy as np
import statsmodels.api as sm
import statsmodels.discrete.count_model as smc
import statsmodels.discrete.discrete_model as smd
import statsmodels.formula.api as smf
from scipy.misc import factorial
from sklearn import preprocessing
from statsmodels.base.model import GenericLikelihoodModel

from .regression_models import RegressionBase, build_endog_exog


class HurdleBase(RegressionBase):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularizer_weight=None, normalize_params=False, t_k=None, add_exposure=False):
        # super().__init__()

        super().__init__(response_var, covariates, log_covariates, log_correction, log_correction_const,
                         regularizer_weight, normalize_params, t_k, add_exposure)
        self.response_var = response_var
        self.covariates = covariates
        self.log_covariates = log_covariates
        self.log_correction = log_correction
        self.log_correction_const = log_correction_const
        self.regularizer_weight = regularizer_weight
        self.normalize_params = normalize_params

        self.inputs = self.covariates + self.log_covariates
        self.variables = [self.response_var] + self.inputs

        self.mean = np.nan
        self.std = np.nan

        self.fit_result_inflated = None
        self.fit_result_positive = None

        if self.normalize_params:
            self.scaler = preprocessing.StandardScaler()

        self.inflated_formula = self.build_formula('np.int32(%s==0)' % self.response_var)
        self.positive_formula = self.build_formula(self.response_var)

    def fit(self, X, y=None):
        X = X[self.variables].copy()  # returns a numpy array

        """
        if self.log_correction == 'add':
            log_corr = lambda x: np.log(x+self.log_correction_const)
        elif self.log_correction == 'max':
            log_corr = lambda x: np.log(np.maximum(x, self.log_correction_const))

        if self.log_covariates:
            X[self.log_covariates] = log_corr(X[self.log_covariates])
        """

        if self.normalize_params:
            X[self.inputs] = self.scaler.fit_transform(X[self.inputs].values)

        inflated_model = self.build_inflated_model(X)

        if self.regularizer_weight is None:
            self.fit_result_inflated = inflated_model.fit()
        else:
            self.fit_result_inflated = inflated_model.fit_regularized(alpha=self.regularizer_weight)

        positive_model = self.build_positive_model(X)

        if self.regularizer_weight is None:
            self.fit_result_positive = positive_model.fit()
        else:
            self.fit_result_positive = positive_model.fit_regularized(alpha=self.regularizer_weight)

        return self

    def predict(self, X, choice=None):
        X = X[self.variables].copy()

        if self.log_correction == 'add':
            def log_corr(x):
                return np.log(x + self.log_correction_const)
        elif self.log_correction == 'max':
            def log_corr(x):
                return np.log(np.maximum(x, self.log_correction_const))

        if self.log_covariates:
            X[self.log_covariates] = log_corr(X[self.log_covariates])

        if self.normalize_params:
            X[self.inputs] = self.scaler.transform(X[self.inputs].values)

        pi = (1 - self.fit_result_inflated.predict(X))
        lam = self.fit_result_positive.predict(X)

        if choice is None:
            return pi * lam

        elif choice == 'all':
            return pi * lam, pi, lam
        else:
            return ValueError('Invalid value for choice')


class TruncatedPoisson(GenericLikelihoodModel):
    def hessian_factor(self, params, scale=None, observed=True):
        pass

    def information(self, params):
        pass

    def nloglikeobs(self, params):
        XB = np.dot(self.exog, params)
        endog = self.endog
        return -endog * XB + np.log(np.exp(np.exp(XB)) - 1) + np.log(factorial(endog))

    def predict(self, params, exog=None, *args, **kwargs):
        if exog is None:
            exog = self.exog

        fitted = np.dot(exog, params[:exog.shape[1]])
        return np.exp(fitted)  # not cdf


"""
class PoissonHurdleRegression(HurdleBase):
    def build_inflated_model(self, X):
        return smd.Logit.from_formula(formula=self.inflated_formula, data=X)

    def build_positive_model(self, X):
        X = X[X[self.response_var]>0]
        endog, exog, (mean, std)= build_endog_exog(X, self.response_var, self.covariates, self.log_covariates, 
                self.log_correction, self.log_correction_const)

        self.mean = mean
        self.std = std

        return TruncatedPoisson(endog=endog, exog=exog)
        #return smd.Poisson.from_formula(formula=self.positive_formula, data=X)

    def predict(self, X, choice=None):
        #X = (X - self.mean) / self.std

        endog, exog, (mean, std)= build_endog_exog(X, self.response_var, self.covariates, self.log_covariates, 
                self.log_correction, self.log_correction_const, self.mean, self.std)

        return (1 - self.fit_result_inflated.predict(X)) * self.fit_result_positive.predict(exog=exog)
"""


class PoissonHurdleRegression(HurdleBase):
    def build_inflated_model(self, X):
        return smd.Logit.from_formula(formula=self.inflated_formula, data=X)

    def build_positive_model(self, X):
        X = X[self.variables].copy()
        X = X[X[self.response_var] > 0]
        X[self.response_var][:] -= 1

        return smf.glm(self.positive_formula, data=X, family=sm.genmod.families.family.Poisson())

    def predict(self, X, choice=None):
        X = X[self.variables].copy()

        """
        if self.log_correction == 'add':
            log_corr = lambda x: np.log(x+self.log_correction_const)
        elif self.log_correction == 'max':
            log_corr = lambda x: np.log(np.maximum(x, self.log_correction_const))

        if self.log_covariates:
            X[self.log_covariates] = log_corr(X[self.log_covariates])
        """

        # if self.normalize_params:
        #    X[self.inputs] = self.scaler.transform(X[self.inputs].values)

        pi = (1 - self.fit_result_inflated.predict(X))
        lam = self.fit_result_positive.predict(X) + 1

        if choice is None:
            return pi * lam

        elif choice == 'all':
            return pi * lam, pi, lam
        else:
            raise ValueError('Invalid value for choice')


class PoissonHurdleFloorRegression(PoissonHurdleRegression):
    def predict(self, X, choice=None):
        pred = super().predict(X, choice)

        return np.floor(pred)


class NegativeBinomialHurdleRegression(PoissonHurdleRegression):
    def build_positive_model(self, X):
        X = X[self.variables].copy()
        X = X[X[self.response_var] > 0]
        X[self.response_var][:] -= 1

        alpha = 2
        print('Alpha=%f' % alpha)
        return smf.glm(self.positive_formula, data=X, family=sm.genmod.families.family.NegativeBinomial(alpha=alpha))


class NegativeBinomialHurdleRegression2(PoissonHurdleRegression):
    def build_positive_model(self, X):
        X = X[self.variables].copy()
        X = X[X[self.response_var] > 0]
        X[self.response_var][:] -= 1

        log_likelihood = 'nb1'
        print('log_likelihood=%s' % log_likelihood)
        return smd.NegativeBinomial.from_formula(self.positive_formula, data=X, loglike_method=log_likelihood)
        # return smf.glm(self.positive_formula, data=X, family=sm.genmod.families.family.NegativeBinomial(alpha=alpha))


class ZeroInflatedPoissonRegression(RegressionBase):
    def build_model(self, X):
        endog, exog, (mean, std) = build_endog_exog(X, self.response_var, self.covariates, self.log_covariates,
                                                    self.log_correction, self.log_correction_const)

        self.mean = mean
        self.std = std

        return smc.ZeroInflatedPoisson(endog, exog=exog, exog_infl=exog)

    def predict(self, X, shape=None):
        _, exog, (mean, std) = build_endog_exog(X, self.response_var, self.covariates, self.log_covariates,
                                                self.log_correction, self.log_correction_const, self.mean, self.std)
        return self.fit_result.predict(exog=exog, exog_infl=exog)
