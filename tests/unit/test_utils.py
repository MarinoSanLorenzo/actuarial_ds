import pytest
import math
import random
import numpy as np
import pandas as pd
from collections import namedtuple
from src.utils import fit_regularized_model
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance

@pytest.fixture
def dataset() -> pd.DataFrame:
    X,y = load_diabetes(return_X_y=True, as_frame=True)
    df = pd.concat([X,y], axis=1)
    df['exposure'] = np.random.uniform(0,1, X.shape[0])
    return df

class TestUtils:

    def test_run_elastic_net_search(self, dataset:pd.DataFrame):
        formula = f'target ~ ' + ' + '.join( [c for c in df.columns if c not in ("exposure", "target")])
        model = fit_regularized_model(formula=formula,
                              data=dataset,
                              exposure=dataset['exposure'],
                              family=sm.families.Poisson(link=sm.families.links.log()),
                              alpha=1,
                              L1_wt=0.5
                              )
        loss_function = mean_poisson_deviance
        alpha, l1 = 1, 0.5
        hyperparams_space = {'alpha': [0.5, 0.6], 'L1_wt': [0, 0.5, 1]}
        random_hyper_param = {hyperparam: random.choice(range_) for hyperparam, range_ in hyperparams_space.items()}

        # creating folds
        KFOLDS=5
        shuffled_data = dataset.sample(frac=1)
        shuffled_data.reset_index(inplace=True)

        folds = []
        start_idx, step_length = 0, math.floor(shuffled_data.shape[0]/KFOLDS)
        for kfold_nb in range(KFOLDS):
            end_idx = step_length*kfold_nb
            folds.append(shuffled_data[start_idx:end_idx])
            start_idx = end_idx

        ValidationFold = namedtuple('ValidationFold', 'train validation name')
        validation_folds = []
        for val_fold_nb in range(KFOLDS):
            train = pd.concat([fold for nb_fold, fold in enumerate(folds) if nb_fold != val_fold_nb], axis=0)
            validation_folds.append(ValidationFold(train, folds[val_fold_nb], f'val_fold_{val_fold_nb}'))


        for validation_fold in validation_folds:
            model = fit_regularized_model(formula=formula,
                                          data=validation_fold.train,
                                          exposure=validation_fold.train['exposure'],
                                          family=sm.families.Poisson(link=sm.families.links.log()),
                                          alpha=1,
                                          L1_wt=0.5
                                          )
            y_pred, y_true = model.predict(validation_fold.validation), validation_fold.validation['target']
            loss = loss_function(y_true, y_pred)
            #