"""
Helper functions for building models.
"""

from models import active_ignition_grid as aig
from models import autoregressive_grid as ag
from models import bias_grid as bg
from models import grid_models as grid
from models import grid_predictor as gp
from models import linear_regression_grid as lrg
from models import poisson_hurdle_grid as prhg
from models import poisson_regression as pr
from models import poisson_regression_grid as prg
from models import poisson_regression_zero_inflated as pzip
from models import zero_inflated_models as zim
from models import zip_regression_grid as pzip_grid

"""
Baselines
"""


def only_zero_model(covariates):
    model = aig.ActiveIgnitionGridModel(None, None)

    return model


def no_ignition_model_poisson(covariates, bounding_box):
    afm = gp.GridPredictorModel(pr.PoissonRegressionModel(covariates), bounding_box)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


"""
Cluster Models
"""


def only_bias_grid_model(covariates):
    igm = bg.BiasGridModel()
    model = aig.ActiveIgnitionGridModel(None, igm)

    return model


def auto_grid_model(covariates):
    igm = ag.AutoregressiveGridModel()
    model = aig.ActiveIgnitionGridModel(None, igm)

    return model


def no_ignition_model_poisson_zip(covariates, bounding_box):
    afm = gp.GridPredictorModel(pzip.PoissonRegressionZeroInflatedModel(covariates), bounding_box)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


"""
Grid Models
"""


# Grid models
def only_zero_grid_model(covariates):
    model = aig.ActiveIgnitionGridModel(None, None)

    return model


def no_ignition_grid_model_poisson(covariates):
    afm = prg.PoissonRegressionGridModel(covariates)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


def separate_grid_model_poisson(covariates, active='active', regularizer_weight=None, log_shift=.5,
                                log_correction='max'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    def ign_filter_func(x):
        return x[~x[active]]

    def ign_pred_func(x, y):
        return y * (~x[active])

    afm = prg.PoissonRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction, filter_func,
                                         pred_func)
    igm = prg.PoissonRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction, ign_filter_func,
                                         ign_pred_func)
    model = aig.ActiveIgnitionGridModel(afm, igm)

    return model


def separate_grid_model_linear(covariates, active='active', regularizer_weight=None, log_shift=.5,
                               log_correction='max'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    def ign_filter_func(x):
        return x[~x[active]]

    def ign_pred_func(x, y):
        return y * (~x[active])

    afm = lrg.LinearRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction, filter_func,
                                        pred_func)
    igm = lrg.LinearRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction, ign_filter_func,
                                        ign_pred_func)
    model = aig.ActiveIgnitionGridModel(afm, igm)

    return model


def joint_grid_model_linear(covariates, active='active', regularizer_weight=None, log_shift=.5, log_correction='max'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    afm = lrg.LinearRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction, filter_func,
                                        pred_func)
    model = grid.UnifiedGrid(afm)

    return model


def active_only_grid_model_poisson(covariates, active='active', regularizer_weight=None, log_shift=.5,
                                   log_correction='max'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    afm = prg.PoissonRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction, filter_func,
                                         pred_func)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


def active_only_grid_model_hurdle(covariates, active='active', regularizer_weight=None, log_shift=.5,
                                  log_correction='max'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    afm = zim.PoissonHurdleRegression(covariates, regularizer_weight, log_shift, log_correction, filter_func, pred_func)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


def joint_grid_model_poisson(covariates, active='active', regularizer_weight=None, log_shift=.5, log_correction='max'):
    afm = prg.PoissonRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction)
    model = grid.UnifiedGrid(afm)

    return model


def active_only_grid_model_poisson_hurdle(covariates, active='active', regularizer_weight=None, log_shift=1,
                                          log_correction='add'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    afm = prhg.PoissonRegressionHurdleGridModel(covariates, regularizer_weight, log_shift, log_correction, filter_func,
                                                pred_func)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


def active_only_grid_model_zip(covariates, active='active', regularizer_weight=None, log_shift=1, log_correction='add'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    afm = pzip_grid.ZIPRegressionGridModel(covariates, regularizer_weight, log_shift, log_correction, filter_func,
                                           pred_func)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


def active_only_grid_model_linear(covariates, active='active', regularizer_weight=None, log_shift=1,
                                  log_correction='add'):
    def filter_func(x):
        return x[x[active]]

    def pred_func(x, y):
        return y * x[active]

    afm = lrg.LinearRegressionGridModel(covariates, regularizer_weight, filter_func, pred_func)
    model = aig.ActiveIgnitionGridModel(afm, None)

    return model


def active_ig_grid_model_poisson(covariates):
    def filter_func(x):
        return x[x[x.active]]

    def pred_func(x, y):
        return y * x[x.active]

    afm = prg.PoissonRegressionGridModel(covariates, filter_func, pred_func)

    def filter_func(x):
        return x[not x.active]

    def pred_func(x, y):
        return y * (not x.active)

    ifm = prg.PoissonRegressionGridModel(covariates, filter_func, pred_func)

    model = aig.ActiveIgnitionGridModel(afm, ifm)

    return model
