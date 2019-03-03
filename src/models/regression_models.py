"""
Regression models with different distributional assumptions.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
import statsmodels.formula.api as smf
import statsmodels.genmod.families.links
from sklearn import preprocessing

from .base.model import Model
from .grid_models import ActiveIgnitionGrid, SwitchingRegressionGrid


def build_endog_exog(X, response_var, covariates, log_covariates, log_correction, log_correction_const, mean=None,
                     std=None):
    endog = np.zeros(len(X))
    endog[:] = X[response_var]

    exog = np.zeros((len(X), len(covariates + log_covariates) + 1))
    for i, cov in enumerate(covariates):
        exog[:, i] = X[cov]

    for i, cov in enumerate(log_covariates):
        if log_correction == 'max':
            exog[:, i + len(covariates)] = np.log(np.maximum(X[cov], log_correction_const))
        elif log_correction == 'add':
            exog[:, i + len(covariates)] = np.log(X[cov] + log_correction_const)

    exog[:, -1] = 1

    if (mean is None) or (std is None):
        mean = np.mean(exog[:, :-1], axis=0)
        std = np.std(exog[:, :-1], axis=0)
        std[std < 1e-5] = 1

    exog[:, :-1] = (exog[:, :-1] - mean) / std

    return endog, exog, (mean, std)


class BasicModelBase(Model):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None, add_exposure=False):
        super().__init__()
        self.response_var = response_var
        self.covariates = covariates
        self.log_covariates = log_covariates
        self.log_correction = log_correction
        self.log_correction_const = log_correction_const
        self.regularization_weight = regularization_weight
        self.normalize_params = normalize_params
        self.t_k = t_k
        self.add_exposure = add_exposure

    def predict(self, X, shape=None):
        pass

    def fit(self, X, y):
        pass


class MeanModel(BasicModelBase):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None, add_exposure=False):
        super().__init__(response_var, covariates, log_covariates, log_correction, log_correction_const,
                         regularization_weight, normalize_params, t_k, add_exposure)
        self.mean = None

    def fit(self, X, y=None):
        self.mean = np.mean(X[self.response_var])
        print(np.shape(X))

        return self

    def predict(self, X, choice=None):
        print(np.shape(X))
        return np.full(X['num_det'].shape, self.mean)


class RegressionBase(BasicModelBase):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None, add_exposure=False):
        super().__init__(response_var, covariates, log_covariates, log_correction, log_correction_const,
                         regularization_weight, normalize_params, t_k, add_exposure)

        self.inputs = self.covariates + self.log_covariates
        self.variables = [self.response_var] + self.inputs
        self.exposure = None
        self.mean = None
        self.std = None

        if self.normalize_params:
            self.scaler = preprocessing.StandardScaler()

        self.fit_result = None
        self.formula = self.build_formula(self.response_var)

    def build_model(self, X):
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

        """
        elif self.log_covariates:
            formula += ' + ' + ' + '.join(self.log_covariates)
        """

        if self.log_covariates and self.log_correction == 'max' and self.log_correction_const == 0:
            log_covariates = list(map(lambda x: log_fmt % (x, x), self.log_covariates))
            # log_covariates = list(map(lambda x: log_fmt % (x), self.log_covariates))
            formula += ' + ' + ' + '.join(log_covariates)

        elif self.log_covariates:
            log_covariates = list(map(lambda x: log_fmt % (x, self.log_correction_const), self.log_covariates))
            formula += ' + ' + ' + '.join(log_covariates)

        print(formula)

        return formula

    def fit(self, X, y=None):
        """
        :param X: covariate dataframe
        :param y: currently unused
        """
        # X = X.filter(self.variables, axis=1)

        # self.mean = X[self.inputs].mean(numeric_only=True)
        # self.std = X[self.inputs].std(numeric_only=True)

        # If std. dev. is too small, don't want to divide by it
        # self.std[self.std<1e-5] = 1

        # X[self.inputs] = (X[self.inputs] - self.mean) / self.std
        if self.add_exposure:
            self.exposure = X.exposure.values
        else:
            self.exposure = None

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

        model = self.build_model(X)

        # TODO: Does fit_regularized() with alpha=0 behave the same as fit()?
        if self.regularization_weight is None:
            self.fit_result = model.fit()
        else:
            self.fit_result = model.fit_regularized(alpha=self.regularization_weight)

        return self

    def predict(self, X, choice=None):
        # X = X.filter(self.variables, axis=1)
        # X[self.inputs] = (X[self.inputs] - self.mean) / self.std

        X = X[self.variables].copy()

        """
        if self.log_correction == 'add':
            log_corr = lambda x: np.log(x+self.log_correction_const)
        elif self.log_correction == 'max':
            log_corr = lambda x: np.log(np.maximum(x, self.log_correction_const))

        if self.log_covariates:
            X[self.log_covariates] = log_corr(X[self.log_covariates])
        """

        if self.normalize_params:
            X[self.inputs] = self.scaler.transform(X[self.inputs].values)

        pred = self.fit_result.predict(X)

        if choice is None:
            return pred
        elif choice == 'all':
            return pred, np.full_like(pred, np.nan)
        else:
            raise ValueError('Invalid value for choice')

    def get_model(self):
        return self.fit_result


class GLMRegression(RegressionBase):
    @property
    def family(self):
        raise NotImplementedError()

    def build_model(self, X):
        return smf.glm(self.formula, data=X, family=self.family, exposure=self.exposure)


class LinearRegression(RegressionBase):
    def build_model(self, X):
        return smf.ols(formula=self.formula, data=X)


class PoissonRegression(GLMRegression):
    family = sm.genmod.families.family.Poisson()


class PoissonModeRegression(GLMRegression):
    family = sm.genmod.families.family.Poisson()

    def predict(self, X, choice=None):
        # X = X.filter(self.variables, axis=1)
        # X[self.inputs] = (X[self.inputs] - self.mean) / self.std

        X = X[self.variables].copy()

        """
        if self.log_correction == 'add':
            log_corr = lambda x: np.log(x+self.log_correction_const)
        elif self.log_correction == 'max':
            log_corr = lambda x: np.log(np.maximum(x, self.log_correction_const))

        if self.log_covariates:
            X[self.log_covariates] = log_corr(X[self.log_covariates])
        """

        if self.normalize_params:
            X[self.inputs] = self.scaler.transform(X[self.inputs].values)

        pred = self.fit_result.predict(X)
        pred = np.floor(pred)

        if choice is None:
            return pred
        elif choice == 'all':
            return pred, np.full_like(pred, np.nan)
        else:
            raise ValueError('Invalid value for choice')


class NegativeBinomialRegression(GLMRegression):
    family = sm.genmod.families.family.NegativeBinomial(alpha=.01)


class PoissonRegressionDiff(GLMRegression):
    # noinspection PyUnusedLocal
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None):
        super().__init__('det_diff', covariates, log_covariates, log_correction, log_correction_const,
                         regularization_weight, normalize_params, t_k)

    def predict(self, X, choice=None):
        pred = super().predict(X, choice=None)
        return pred + X['num_det']

    family = sm.genmod.families.family.Poisson()


class LogisticRegression(GLMRegression):
    def build_model(self, X):
        self.formula = self.build_formula('np.int32(%s)' % self.response_var)
        return smd.Logit.from_formula(formula=self.formula, data=X)

    @property
    def family(self):
        raise NotImplementedError()


class LogisticBinaryRegression(GLMRegression):
    def build_model(self, X):
        self.formula = self.build_formula('np.int32(%s>1)' % self.response_var)
        return smd.Logit.from_formula(formula=self.formula, data=X)

    @property
    def family(self):
        raise NotImplementedError()


class LogNormalRegression(GLMRegression):
    family = sm.genmod.families.family.Gaussian(link=sm.genmod.families.links.log)


"""
class NegativeBinomialRegression(RegressionBase):
    def build_model(self, X):
        endog, exog, (mean, std)= build_endog_exog(X, self.response_var, self.covariates, self.log_covariates, 
                self.log_correction, self.log_correction_const)

        self.mean = mean
        self.std = std

        return smd.NegativeBinomial(endog=endog, exog=exog)

    def predict(self, X, choice=None):
        _, exog, (mean, std) = build_endog_exog(X, self.response_var, self.covariates, self.log_covariates,
                self.log_correction, self.log_correction_const, self.mean, self.std)
        return self.fit_result.predict(exog=exog)
"""


class PersistenceModel(BasicModelBase):
    def fit(self, X, y=None):
        return self

    def predict(self, X, choice=None):
        return np.array(X['num_det'].values)


class ZeroModel(BasicModelBase):
    def fit(self, X, y=None):
        return self

    def predict(self, X, choice=None):
        return np.zeros(X['num_det'].shape)


class PersistenceAugmented(BasicModelBase):
    def fit(self, X, y=None):
        return self

    def predict(self, X, choice=None):
        today = X['vpd_%d' % self.t_k].values
        grad = (X['vpd'] - today) / today

        grad[np.isnan(grad)] = 0
        grad[grad == np.inf] = 0
        grad[grad == -np.inf] = 0

        grad[grad > 1] = 1

        print(np.min(grad), np.max(grad), np.mean(grad))

        return np.array(X['num_det'].values) * (1 + grad)


class PersistenceAugmentedParam(LinearRegression):
    # noinspection PyUnusedLocal
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None):
        super().__init__('det_diff', covariates, log_covariates, log_correction, log_correction_const,
                         regularization_weight, normalize_params, t_k)

        self.inputs = self.covariates + self.log_covariates
        self.variables = [self.response_var] + self.inputs

        if self.normalize_params:
            self.scaler = preprocessing.StandardScaler()

        self.fit_result = None
        self.formula = self.build_formula(self.response_var)

    def build_formula(self, response_var, remove_intercept=False):
        # TODO: Determine which to use

        """
        if remove_intercept:
            formula = '%s ~ -1' % response_var
        else:
            formula = '%s ~ 1' % response_var

            if self.log_correction == 'add':
                log_fmt = 'np.log(%s+%f)'
            elif self.log_correction == 'max':
                log_fmt = 'np.log(np.maximum(%s, %f))'
            else:
                raise ValueError('Invalid log_correction: %s' % self.log_correction)

            if self.covariates:
                formula += ' + ' + ' + '.join(self.covariates)

            if self.log_covariates:
                formula += ' + ' + ' + '.join(self.log_covariates)
            if self.log_covariates:
                log_covariates = list(map(lambda x: log_fmt % (x, self.log_correction_const), self.log_covariates))
                formula += ' + ' + ' + '.join(log_covariates)

            return formula
        """

        return 'det_diff ~ vpd_grad:num_det - 1'

    def predict(self, X, choice=None):
        # X = X.filter(self.variables, axis=1)
        # X[self.inputs] = (X[self.inputs] - self.mean) / self.std

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
            X[self.log_covariates] = log_corr(X[self.log_covariates])

        if self.normalize_params:
            X[self.inputs] = self.scaler.transform(X[self.inputs].values)

        pred = self.fit_result.predict(X) + X['num_det']

        if choice is None:
            return pred
        elif choice == 'all':
            return pred, np.full_like(pred, np.nan)
        else:
            raise ValueError('Invalid value for choice')


DET_CUTOFF = 5


def large_filter_func(x):
    return x[x.num_det > DET_CUTOFF]


def small_filter_func(x):
    return x[x.num_det <= DET_CUTOFF]


def large_pred_func(x, y):
    return y * (x.num_det > DET_CUTOFF)


def small_pred_func(x, y):
    return y * (x.num_det <= DET_CUTOFF)


class LargeSplitModel(Model):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None):
        super().__init__()
        self.large_model = PoissonRegression(response_var, covariates, log_covariates, log_correction,
                                             log_correction_const, regularization_weight, normalize_params, t_k=t_k)
        self.small_model = PoissonRegression(response_var, covariates, log_covariates, log_correction,
                                             log_correction_const, regularization_weight, normalize_params, t_k=t_k)

        self.model = ActiveIgnitionGrid(self.large_model, self.small_model, large_filter_func, small_filter_func,
                                        large_pred_func, small_pred_func)

    def fit(self, X, y=None):
        self.model.fit([X, X], y)
        return self

    def predict(self, X, shape=None, choice=None):
        return self.model.predict([X, X], shape, choice)


def cum_large_filter_func(x):
    return x[x.large_fire]


def cum_small_filter_func(x):
    return x[~x.large_fire]


def cum_large_pred_func(x, y):
    return y * x.large_fire


def cum_small_pred_func(x, y):
    return y * (~x.large_fire)


class CumulativeLargeSplitPerfectModel(Model):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None):
        super().__init__()
        self.large_model = PoissonRegression(response_var, covariates, log_covariates, log_correction,
                                             log_correction_const, regularization_weight, normalize_params, t_k=t_k)
        self.small_model = PoissonRegression(response_var, covariates, log_covariates, log_correction,
                                             log_correction_const, regularization_weight, normalize_params, t_k=t_k)

        self.model = ActiveIgnitionGrid(self.large_model, self.small_model, cum_large_filter_func,
                                        cum_small_filter_func,
                                        cum_large_pred_func, cum_small_pred_func)

    def fit(self, X, y=None):
        self.model.fit([X, X], y)
        return self

    def predict(self, X, shape=None, choice=None):
        return self.model.predict([X, X], shape, choice)


class LargeFireOracleModel(Model):
    def fit(self, X, y=None):
        return self

    def predict(self, X, shape=None, choice=None):
        return X.large_fire


class CumulativeLargeSplitModel(Model):
    def __init__(self, response_var, covariates, log_covariates, log_correction, log_correction_const,
                 regularization_weight=None, normalize_params=False, t_k=None):
        super().__init__()
        self.large_model = PoissonRegression(response_var, covariates, log_covariates, log_correction,
                                             log_correction_const, regularization_weight, normalize_params, t_k=t_k)
        self.small_model = PoissonRegression(response_var, covariates, log_covariates, log_correction,
                                             log_correction_const, regularization_weight, normalize_params, t_k=t_k)

        # self.mixture_model = LogisticRegression('large_fire', covariates, log_covariates, log_correction,
        #        log_correction_const, regularization_weight, normalize_params, t_k=t_k)

        self.mixture_model = LargeFireOracleModel()

        self.model = SwitchingRegressionGrid(self.large_model, self.small_model, self.mixture_model,
                                             cum_large_filter_func, cum_small_filter_func)

    def fit(self, X, y=None):
        self.model.fit([X, X, X], y)
        return self

    def predict(self, X, shape=None, choice=None):
        return self.model.predict([X, X, X], shape, choice)
