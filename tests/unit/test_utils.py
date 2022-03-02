import pytest
import numpy as np
import pandas as pd
from src.utils import fit_regularized_model
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
import statsmodels.formula.api as smf

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


