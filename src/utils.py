from typing import Dict, Tuple, List, Union
import pandas as pd
from collections import defaultdict

__all__ = ['get_distribution', 'get_distribution_info_for_categorical_variables', 'get_avg_target_per_numerical_bin']


def get_avg_target_per_numerical_bin(df:pd.DataFrame,feature:str, target:Union[str, List[str]]) -> Union[pd.DataFrame, pd.Series]:
    df_cp = df.copy()
    binned_feature = f'{feature}_bin_quantile_based'
    df_cp[binned_feature] = pd.qcut(df[feature], q=10, precision=0, duplicates='drop')
    return df_cp.groupby(binned_feature)[target].mean()


def get_distribution_info_for_categorical_variables(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Construct a dataframe with information about the distribution of the categorical variables
    '''
    info = defaultdict(list)
    cat_data = df.select_dtypes(include=['object'])
    cols = cat_data.columns
    for col in cols:
        nb_unique_values, dist = get_distribution(df[col])
        info['nb_unique_categories'].append(nb_unique_values),  info['distribution'].append(dist)
    return pd.DataFrame(info, index=cols)

def get_distribution(s:pd.Series) -> Tuple[int, Dict[str, str]]:
    '''
    Returns the unique number of categories and an ordered format distribution for categorical variables
    '''
    dist = s.value_counts(dropna=False, normalize=True).to_dict()
    dist_ordered = dict(sorted(dist.items(), key=lambda item: item[1], reverse=True))
    return len(s.unique()), {k: '{:.2%}'.format(v) for k, v in dist_ordered.items()}

