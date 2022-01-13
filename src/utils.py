from typing import Dict, Tuple
import pandas as pd
from collections import defaultdict

__all__ = ['get_distribution', 'get_distribution_info_for_categorical_variables']

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

