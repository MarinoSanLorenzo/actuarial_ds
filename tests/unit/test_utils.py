import pytest
import time
import math
import random
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
from src.utils import fit_regularized_model
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from src.utils import get_validation_folds, run_cross_validation, run_hyperopt

@pytest.fixture
def df() -> pd.DataFrame:
    X,y = load_diabetes(return_X_y=True, as_frame=True)
    df = pd.concat([X,y], axis=1)
    df['exposure'] = np.random.uniform(0,1, X.shape[0])
    return df

class TestUtils:

    def test_run_elastic_net_search(self, df:pd.DataFrame):
        formula = f'target ~ ' + ' + '.join( [c for c in df.columns if c not in ("exposure", "target")])
        loss_function = mean_poisson_deviance
        hyperparams_space = {'alpha': [0.5, 0.6],
                             'L1_wt': [0, 0.5, 1],
                             'formula':[formula],
                             'data':[None],
                             'exposure':[None],
                             'family':[sm.families.Poisson(link=sm.families.links.log())]}
        params_to_record = ['alpha', 'L1_wt']
        cv_results = run_hyperopt(df=df,
                                  hyperparams_space=hyperparams_space,
                                  model_fit_func=fit_regularized_model,
                                  loss_function=loss_function,
                                  params_to_record=params_to_record)

        assert isinstance(cv_results, pd.DataFrame)
        assert 'losses' in cv_results.columns
        assert 'cv_mean_loss' in cv_results.columns
        assert 'cv_std_loss' in cv_results.columns
        assert 'hyperparams' in cv_results.columns
        for param_to_record in params_to_record:
            assert param_to_record in cv_results.columns

