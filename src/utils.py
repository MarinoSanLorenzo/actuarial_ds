from typing import Dict, Tuple, List, Union
import pandas as pd
from collections import defaultdict
import numpy as np

__all__ = ['get_distribution', 'get_distribution_info_for_categorical_variables', 'get_avg_target_per_numerical_bin', 'apply_mean_hot_categorical_encoding', 'bin_numerical_variables', 'get_reference_classes']


def get_reference_classes(df:pd.DataFrame, target:str, exposure_name:str) -> List[str]:
    reference_class_lst = []
    df['claim_frequency'] = df[target] / df[exposure_name]
    for cat_var in df.select_dtypes(include=['category']):
        avg_frequency_per_category = pd.DataFrame(df.groupby(cat_var)['claim_frequency'].mean()).sort_values(
            by=['claim_frequency'], ascending=False)
        reference_class = avg_frequency_per_category.tail(1).last_valid_index()
        print(f'The reference class of {cat_var} is {reference_class}')
        reference_class_lst.append(f'{cat_var}_{reference_class}')
    df.drop(columns='claim_frequency', inplace=True)
    return reference_class_lst

def bin_numerical_variables(df:pd.DataFrame, var_name:str, nb_bin:int=5) -> pd.DataFrame:
    labels = [str(interval) for interval in list(pd.qcut(df[var_name], q=nb_bin).unique().categories)]
    df[f'{var_name}_bin'] = pd.Categorical(pd.qcut(df[var_name], q=nb_bin, labels=labels))
    return df

def apply_mean_hot_categorical_encoding(df:pd.DataFrame, target:str, var_name:str, nb_bin:int=3, risk_groups:list=None) -> Tuple[pd.DataFrame, Dict]:
    avg_frequency = pd.DataFrame(df.groupby(var_name)[target].mean())
    avg_freq_df = pd.DataFrame(pd.qcut(avg_frequency[target], q=nb_bin))
    categories = avg_freq_df[target].unique().categories

    if not risk_groups:
        risk_groups = [f'low', \
                           f'medium', \
                           f'high']
    if nb_bin!= len(risk_groups):
        raise ValueError(f'nb_bin {nb_bin} and lenght risk_groups {len(risk_groups)} should be the same!')

    cond_list = [(avg_frequency[target] > category.left) & (avg_frequency[target] <= (category.right + 0.001)) for category in categories]

    avg_frequency[f'{var_name}_risk_group'] = np.select(cond_list, risk_groups)
    risk_group_mapping = avg_frequency[f'{var_name}_risk_group'].to_dict()
    df[f'{var_name}_risk_group'] = pd.Categorical(df.district.replace(risk_group_mapping))
    return df, risk_group_mapping



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
    cat_data = df.select_dtypes(include=['object', 'category'])
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

