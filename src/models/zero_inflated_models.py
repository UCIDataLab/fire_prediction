import numpy as np
import statsmodels.api as sm
import statsmodels.discrete.count_model as smc
import statsmodels.discrete.discrete_model as smd
import statsmodels.formula.api as smf
from scipy.misc import factorial
from sklearn import preprocessing
from statsmodels.base.model import GenericLikelihoodModel

from .regression_models import RegressionBase, build_endog_exog
from scipy.special import gammaln
from scipy import stats


class HurdleBase(RegressionBase):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None, add_exposure=False):
        # super().__init__()

        super().__init__(response_var, covariates, log_covariates, log_correction, log_correction_const,
                         regularization_weight, normalize_params, t_k, add_exposure)
        self.response_var = response_var
        self.covariates = covariates
        self.log_covariates = log_covariates
        self.log_correction = log_correction
        self.log_correction_const = log_correction_const
        self.regularization_weight = regularization_weight
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

    def build_model(self, X):
        raise NotImplementedError()

    def build_inflated_model(self, X):
        raise NotImplementedError()

    def build_positive_model(self, X):
        raise NotImplementedError()

    def build_formula(self, response_var, remove_intercept=False):
        if remove_intercept:
            formula = '%s ~ -1' % response_var
        else:
            formula = '%s ~ 1' % response_var

        if self.log_correction == 'add':
            log_fmt = 'np.log(%s+%f)'
        elif self.log_correction == 'max' and self.log_correction_const == 0:
            """Automatically learn value for correction"""
            # Assumes value being corrected is discrete (otherwise values under 1 are rounded up)
            y_star = 'np.maximum(%s, 1)'
            d = 'np.mod(np.minimum(%s,1)+1,2)'
            # d = '%s==0'
            log_fmt = 'np.log(%s) + %s' % (y_star, d)
            # log_fmt = 'np.log(%s)' % (y_star)
        elif self.log_correction == 'max':
            log_fmt = 'np.log(np.maximum(%s, %f))'
        else:
            raise ValueError('Invalid log_correction: %s' % self.log_correction)

        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

        if self.log_covariates:
            formula += ' + ' + ' + '.join(self.log_covariates)

        """
        if self.log_covariates and self.log_correction == 'max' and self.log_correction_const == 0:
            log_covariates = list(map(lambda x: log_fmt % (x, x), self.log_covariates))
            # log_covariates = list(map(lambda x: log_fmt % (x), self.log_covariates))
            formula += ' + ' + ' + '.join(log_covariates)

        elif self.log_covariates:
            log_covariates = list(map(lambda x: log_fmt % (x, self.log_correction_const), self.log_covariates))
            formula += ' + ' + ' + '.join(log_covariates)
        """

        print(formula)

        return formula

    def fit(self, X, y=None):
        if self.add_exposure:
            self.exposure = X.exposure.values
        else:
            self.exposure = None

        X = X[self.variables].copy()  # returns a numpy array

        if 'vpd' in self.log_covariates:
            log_corr = lambda x: np.log(x+1)
        elif self.log_correction == 'add':
            log_corr = lambda x: np.log(x+self.log_correction_const)
        elif self.log_correction == 'max':
            log_corr = lambda x: np.log(np.maximum(x, self.log_correction_const))

        if self.log_covariates:
            print('Fit - Log')
            X[self.log_covariates] = log_corr(X[self.log_covariates])

        if self.normalize_params:
            print('Fit - Norm')
            X[self.inputs] = self.scaler.fit_transform(X[self.inputs].values)

        inflated_model = self.build_inflated_model(X)

        if self.regularization_weight is None:
            self.fit_result_inflated = inflated_model.fit()
        else:
            self.fit_result_inflated = inflated_model.fit_regularized(alpha=self.regularization_weight)

        positive_model = self.build_positive_model(X)

        if self.regularization_weight is None:
            self.fit_result_positive = positive_model.fit()
        else:
            self.fit_result_positive = positive_model.fit_regularized(alpha=self.regularization_weight)

        return self

    def predict(self, X, choice=None):
        if self.add_exposure:
            self.exposure = X.exposure.values
        else:
            self.exposure = None

        X = X[self.variables].copy()

        if self.log_correction == 'add':
            def log_corr(x):
                return np.log(x + self.log_correction_const)
        elif self.log_correction == 'max':
            def log_corr(x):
                return np.log(np.maximum(x, self.log_correction_const))
        else:
            raise ValueError("Log Correction value '%s' is invalid." % self.log_correction)

        if self.log_covariates:
            print('Pred - Log')
            X[self.log_covariates] = log_corr(X[self.log_covariates])

        if self.normalize_params:
            print('Pred - Norm')
            X[self.inputs] = self.scaler.transform(X[self.inputs].values)

        pi = (1 - self.fit_result_inflated.predict(X))
        lam = self.fit_result_positive.predict(X, exposure=self.exposure)

        if choice is None:
            return pi * lam

        elif choice == 'all':
            return pi * lam, pi, lam
        else:
            return ValueError('Invalid value for choice')


"""
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


class TruncatedPoisson(smd.Poisson):
    family = None

    def information(self, params):
        raise NotImplementedError()

    def cdf(self, X):
        y = self.endog

        cdf_of_zero = stats.poisson.cdf(0, np.exp(X))

        return (stats.poisson.cdf(y, np.exp(X)) - cdf_of_zero) / (1 - cdf_of_zero)

    def pdf(self, X):
        y = self.endog
        mu = np.exp(X)

        # logpmf = y*np.log(mu) - np.log(np.exp(mu) - 1) - np.log(gammaln(y + 1))
        # logpmf = np.exp(stats.poisson.logpmf(y, mu)) - np.log(1 - np.exp(-mu))
        logpmf = stats.poisson.logpmf(y, mu) - np.log(1 - np.exp(stats.poisson.logpmf(0, mu)))

        return np.exp(logpmf)

    def loglike(self, params):
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog

        # return np.sum(-np.exp(XB) + endog * XB - gammaln(endog + 1))

        # return np.sum(-np.log(np.exp(np.exp(XB)) - 1) + endog * XB - gammaln(endog + 1))

        a = -np.exp(XB) + endog * XB - gammaln(endog + 1)
        b = np.log(1 - np.exp(-np.exp(XB)))

        return np.sum(a - b)

    def loglikeobs(self, params):
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        # return -np.exp(XB) + endog * XB - gammaln(endog + 1)

        # return -np.log(np.exp(np.exp(XB)) - 1) + endog * XB - gammaln(endog + 1)

        a = -np.exp(XB) + endog * XB - gammaln(endog + 1)
        b = np.log(1 - np.exp(-np.exp(XB)))

        return a - b

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        if start_params is None and self.data.const_idx is not None:
            # k_params or k_exog not available?
            start_params = 0.001 * np.ones(self.exog.shape[1])
            start_params[self.data.const_idx] = self._get_start_params_null()[0]

        cntfit = super(smc.CountModel, self).fit(start_params=start_params,
                                             method=method, maxiter=maxiter, full_output=full_output,
                                             disp=disp, callback=callback, **kwargs)

        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type': kwargs['cov_type'], 'cov_kwds': cov_kwds}
        else:
            kwds = {}
        discretefit = smd.PoissonResults(self, cntfit, **kwds)
        return smd.PoissonResultsWrapper(discretefit)

    def score(self, params):
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)

        # return np.dot(self.endog - L, X)

        # R = np.exp(L + np.dot(X, params) + offset + exposure) / (np.exp(L) - 1)
        # return np.dot(self.endog - R, X)

        R = L / (np.exp(L) - 1)

        return np.dot(self.endog - L - R, X)

    def score_obs(self, params):
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)
        # return (self.endog - L)[:, None] * X

        # R = np.exp(L + np.dot(X, params) + offset + exposure) / (np.exp(L) - 1)
        # return self.endog - R[:, None] * X

        R = L / (np.exp(L) - 1)

        return (self.endog - L - R)[:,None] * X

    def score_factor(self, params):
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)

        # return (self.endog - L)

        # R = np.exp(L + np.dot(X, params) + offset + exposure) / (np.exp(L) - 1)
        # return self.endog - R

        R = L / (np.exp(L) - 1)

        return self.endog - L - R

    def hessian(self, params):
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + exposure + offset)
        # return -np.dot(L * X.T, X)

        a = L / (np.exp(L) - 1)
        b = (L**2 * np.exp(L)) / (np.exp(L) - 1)**2
        return -np.dot((L + a - b) * X.T, X)

    def hessian_factor(self, params):
        raise NotImplementedError()
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + exposure + offset)

        a = L / (np.exp(L) - 1)
        b = (L**2 * np.exp(L)) / (np.exp(L) - 1)**2

        return L + a - b


class PoissonHurdleRegression(HurdleBase):
    def build_model(self, X):
        raise NotImplementedError()

    def build_inflated_model(self, X):
        return smd.Logit.from_formula(formula=self.inflated_formula, data=X)

    def build_positive_model(self, X):
        X = X[self.variables].copy()
        ind = X[self.response_var] > 0
        X = X[ind]
        # X[self.response_var][:] -= 1

        # return smf.glm(self.positive_formula, data=X, family=sm.genmod.families.family.Poisson())
        # return smc.Poisson.from_formula(self.positive_formula, data=X)
        if self.exposure is not None:
            exposure = self.exposure[ind]
            print('=== Exposure ===', exposure)
            print('shape', exposure.shape, X.shape)
        else:
            exposure = None
        return TruncatedPoisson.from_formula(self.positive_formula, data=X, exposure=exposure)

    # def predict(self, X, choice=None):
    #     X = X[self.variables].copy()
    #
    #     """
    #     if self.log_correction == 'add':
    #         log_corr = lambda x: np.log(x+self.log_correction_const)
    #     elif self.log_correction == 'max':
    #         log_corr = lambda x: np.log(np.maximum(x, self.log_correction_const))
    #
    #     if self.log_covariates:
    #         X[self.log_covariates] = log_corr(X[self.log_covariates])
    #     """
    #
    #     # if self.normalize_params:
    #     #    X[self.inputs] = self.scaler.transform(X[self.inputs].values)
    #
    #     pi = (1 - self.fit_result_inflated.predict(X))
    #     # lam = self.fit_result_positive.predict(X) + 1
    #     lam = self.fit_result_positive.predict(X)
    #
    #     if choice is None:
    #         return pi * lam
    #
    #     elif choice == 'all':
    #         return pi * lam, pi, lam
    #     else:
    #         raise ValueError('Invalid value for choice')


class PoissonHurdleFloorRegression(PoissonHurdleRegression):
    def build_model(self, X):
        raise NotImplementedError()

    def predict(self, X, choice=None):
        pred = super().predict(X, choice)

        return np.floor(pred)


class NegativeBinomialHurdleRegression(PoissonHurdleRegression):
    def build_model(self, X):
        raise NotImplementedError()

    def build_positive_model(self, X):
        X = X[self.variables].copy()
        X = X[X[self.response_var] > 0]
        X[self.response_var][:] -= 1

        alpha = 2
        print('Alpha=%f' % alpha)
        return smf.glm(self.positive_formula, data=X, family=sm.genmod.families.family.NegativeBinomial(alpha=alpha))


class NegativeBinomialHurdleRegression2(PoissonHurdleRegression):
    def build_model(self, X):
        raise NotImplementedError()

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
        _, exog, (_, _) = build_endog_exog(X, self.response_var, self.covariates, self.log_covariates,
                                           self.log_correction, self.log_correction_const, self.mean, self.std)
        return self.fit_result.predict(exog=exog, exog_infl=exog)
