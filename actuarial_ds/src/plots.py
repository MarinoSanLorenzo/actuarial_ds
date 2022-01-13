import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.constants import Constants, params
from typing import List, Dict, Any

__all__ = ['plot_univariate_numerical_variables_distribution', 'plot_univariate_categorical_variables_distribution']


def plot_univariate_categorical_variables_distribution(df:pd.DataFrame, categorical_variables:List[str], params:Dict[str, Any]) -> None:
    for cat_var in categorical_variables:
        if cat_var not in params.get(Constants.VARIABLES_TO_EXCLUDE):
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            dist = df[cat_var].value_counts().head(10) / df.shape[0]
            idx, values = list(dist.index), list(dist.values)
            ax.bar(idx, values)
            plt.title(f'Bin distribution of {cat_var}')
            plt.xlabel(f'Categories of {cat_var}')
            plt.ylabel('Value count')
            plt.show()
            plt.close()

def plot_univariate_numerical_variables_distribution(df:pd.DataFrame, numerical_variables:List[str], params:Dict[str, Any]) -> None:
    for num_var in numerical_variables:
        if num_var not in params.get(Constants.VARIABLES_TO_EXCLUDE):
            sns.distplot(df[num_var])
            plt.show()
            plt.close()