from typing import Dict, Tuple, List, Union, Any, Iterable, NamedTuple, Set
from types import FunctionType
import pandas as pd
from collections import defaultdict, namedtuple
import numpy as np
import statsmodels.genmod.generalized_linear_model

from src.constants import Constants, params
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
import math
import sklearn
import time
import random
from collections import OrderedDict, defaultdict
import re

__all__ = [
    "get_distribution",
    "get_distribution_info_for_categorical_variables",
    "get_avg_target_per_numerical_bin",
    "apply_mean_hot_categorical_encoding",
    "bin_numerical_variables",
    "get_reference_classes",
    "run_glm_backward_selection",
    'fit_regularized_model',
    'run_hyperparameter_search_regularized_model',
    'get_validation_folds',
    'run_cross_validation',
    'run_hyperopt',
    'get_features_modalities',
    'get_nb_max_combinations',
    'select_random_modality_vector',
    'generate_all_profiles',
    'does_profile_belong_to_ref_class',
    'PurePremiunProfile',
    'get_risk_premium',
    'does_profile_belong_to_ref_class',
    'get_all_pure_premium_profiles'

]

ValidationFold = namedtuple('ValidationFold', 'train validation name')




class PurePremiunProfile(NamedTuple):
    profile_name: str
    features_lst: List[str]
    mean_freq: float
    mean_sev: float
    pure_premium_risk: float
    calc_details:str

def get_all_pure_premium_profiles(features_modalities:Dict[str, List[str]],
                              exp_params_freq:Dict[str, float],
                              exp_params_sev:Dict[str, float],
                              reference_class_lst:List[str],
                              reference_class_sev_lst:List[str],
                              sep_token:str='---') -> Dict[str, PurePremiunProfile]:
    unique_profiles = generate_all_profiles(features_modalities, sep_token=sep_token)
    premium_profiles = defaultdict(list)
    for profile in unique_profiles:
        profile_parsed = profile.split(sep_token)
        pp_freq = get_risk_premium(profile_parsed, exp_params_freq, reference_class_lst)
        pp_sev = get_risk_premium(profile_parsed, exp_params_sev, reference_class_sev_lst)
        mean_freq, mean_sev = pp_freq.risk_premium, pp_sev.risk_premium
        premium_profiles[profile] = PurePremiunProfile(
            profile_name=profile,
            features_lst=profile_parsed,
            mean_freq=mean_freq,
            mean_sev=mean_sev,
            pure_premium_risk=mean_freq * mean_sev,
            calc_details = f'{pp_freq.formula_with_num_details}*{pp_sev.formula_with_num_details}'
        )
    return premium_profiles

class RiskPremiumCalculationDetail(NamedTuple):
    risk_premium:float
    formula_with_num_details:str

def get_risk_premium(profile: List[str], exp_params_risk: Dict[str, float], reference_class: List[str]) -> float:
    risk_premium = exp_params_risk.get('Intercept')
    formula_with_num_details = f'(exp(Intercept)={round(risk_premium,2)})'

    if does_profile_belong_to_ref_class(profile, reference_class):
        return RiskPremiumCalculationDetail(risk_premium=risk_premium,
                                            formula_with_num_details=formula_with_num_details)
    formula_with_num_details_lst = [formula_with_num_details]
    for modality in profile:
        exp_param_risk = exp_params_risk.get(modality, 1)
        risk_premium *= exp_param_risk
        formula_with_num_details = f'(exp({modality})={round(exp_param_risk, 2)})'
        formula_with_num_details_lst.append(formula_with_num_details)

    formula_with_num_details = '*'.join(formula_with_num_details_lst)
    return RiskPremiumCalculationDetail(risk_premium=risk_premium,
                                            formula_with_num_details=formula_with_num_details)

def does_profile_belong_to_ref_class(profile:List[str], reference_class_lst:List[str]) -> bool:
    return all([modality in reference_class_lst for modality in profile])

def generate_all_profiles(features_modallities:Dict[str, List[str]], sep_token:str) -> Set[str]:
    nb_max_combinations = get_nb_max_combinations(features_modallities)
    unique_profiles = set()
    while len(unique_profiles) != nb_max_combinations:
        modality_vector = select_random_modality_vector(features_modallities)
        modality_vector_str = sep_token.join(modality_vector)
        if modality_vector_str not in unique_profiles:
            unique_profiles.add(modality_vector_str)
    return unique_profiles

def select_random_modality_vector(features_modallities:Dict[str, List[str]]) -> List[str]:
    return [random.choice(modality_lst) for modality_lst in features_modallities.values()]

def get_nb_max_combinations(features_modallities:Dict[str, List[str]]) -> int:
    nb_max_combinations = 1
    for v in features_modallities.values():
        nb_max_combinations *= len(v)
    return nb_max_combinations

def get_feature_from_modality(feature_modality:str):
    return feature_modality[:re.search('\_', feature_modality).start()]

def get_features_modalities(scorecard:pd.DataFrame, reference_class_lst:List[str],
                             reference_class_sev_lst:List[str]):
    reference_class_set = set(reference_class_lst) | set(reference_class_sev_lst)
    all_features_modalities = set(set(scorecard.index) | reference_class_set) - set(['Intercept'])
    features_modalities = defaultdict(list)
    for feature_modality in all_features_modalities:
        feature = get_feature_from_modality(feature_modality)
        features_modalities[feature].append(feature_modality)

    lst_features_not_in_model = []
    for feature in features_modalities:
        is_feature_in_model = False
        for modality in set(scorecard.index):
            if feature in modality:
                is_feature_in_model = True
                break
        else:
            lst_features_not_in_model.append(feature)

    for feature_not_in_model in lst_features_not_in_model:
        del features_modalities[feature_not_in_model]

    return OrderedDict(features_modalities)

def run_hyperopt(df:pd.DataFrame,
                 hyperparams_space:Dict[str, Iterable],
                 model_fit_func:FunctionType,
                 loss_function:Union[FunctionType],
                 params_to_record:List[str]=None,
                 limit_time:int=100,
                 max_iter:int=100,
                 exposure_name:str='exposure',
                 target_name:str='target',
                 is_debug:bool=False) -> pd.DataFrame:
    '''
    :param df: Train Dataset on which the data fold will be build
    :param hyperparams_space: dictionarry with the name (that the model_fit_func takes) of the hyperparam and an iterable containing all values to sample from
    :param model_fit_func:  a custom/user-defined wrapper function fitting a model
    :param loss_function:  a user-defined function or sklearn.metric to assess the performance on the k-fold
    :param params_to_record: a list containing all the hyperparam name to record
    :param limit_time: maximum time in seconds to run the hyperopt
    :param max_iter: number of max iterations to run the hyperopt
    :return:
    '''
    max_combinations = np.cumprod([len(v) for v in hyperparams_space.values()])[-1]
    tried_hyperparam = set()
    if not max_iter:
        max_iter = max_combinations
    else:
        max_iter = min(max_iter, max_combinations)
    if is_debug:
        print(f'the number of the maximum iterations is {max_iter}')
    nb_iter = 1
    start = time.time()
    duration = time.time() - start
    cv_results = defaultdict(list)
    while (nb_iter <= max_iter) and (duration < limit_time):
        if is_debug:
            print(f'--------------------------------')
            print(f'{nb_iter} iteration starting...')
            print(f'{duration} time elapsed...')
        random_hyper_param = {hyperparam: random.choice(range_) for hyperparam, range_ in hyperparams_space.items()}
        hyper_space_combi = '-'.join([str(f'{p}:{random_hyper_param.get(p)}') for p in params_to_record])
        if hyper_space_combi in tried_hyperparam:  # do not try an aleady tested hyperparam combination
            continue
        cv_result = run_cross_validation(df=df,
                                         model_fit_func=model_fit_func,
                                         params_models=random_hyper_param,
                                         loss_function=loss_function,
                                         params_to_record=params_to_record,
                                         exposure_name=exposure_name,
                                         target_name=target_name)

        cv_results['losses'].append(cv_result['loss'])
        cv_results['cv_mean_loss'].append(np.mean(cv_result['loss']))
        cv_results['cv_std_loss'].append(np.std(cv_result['loss']))
        cv_results['hyperparams'].append(hyper_space_combi)
        for param_to_record in params_to_record:
            cv_results[param_to_record].append(random_hyper_param[param_to_record])

        tried_hyperparam.add(hyper_space_combi)
        nb_iter += 1
        duration = time.time() - start

    #TODO: return the best score based on the error quadratic lower mean loss and lower std deviation
    return pd.DataFrame.from_dict(cv_results)


def run_cross_validation(df:pd.DataFrame,
                         model_fit_func:FunctionType,
                         params_models:Dict[str,Iterable],
                         loss_function:Union[FunctionType],
                         params_to_record:List[str]=None,
                         exposure_name:str='exposure',
                         target_name:str='target') -> Dict[str, List]:
    '''
    :param df: Train Dataset on which the data fold will be build
    :param model_fit_func: a custom/user-defined wrapper function fitting a model
    :param params_models: a dictionary with the keys as parameter of the model_fit_func
    :param loss_function: a user-defined function or sklearn.metric to assess the performance on the k-fold
    :param params_to_record: a list containing all the hyperparam name to record
    :return: a Dictionary with hyperparameters values, loss and models
    '''

    if not params_to_record:
        params_to_record = list(params_models.keys())
    validation_folds = get_validation_folds(df)
    results = defaultdict(list)

    for validation_fold in validation_folds:
        train_data, exposure = validation_fold.train, validation_fold.train[exposure_name]
        params_models['data'], params_models['exposure'] = train_data, exposure
        model = model_fit_func(**params_models)
        y_pred, y_true = model.predict(validation_fold.validation), validation_fold.validation[target_name]
        loss = loss_function(y_true, y_pred)
        results['loss'].append(loss)
        results['model'].append(model)
        for param_name in params_to_record:
            results[param_name].append(params_models.get(param_name))
    return results


def get_validation_folds(df:pd.DataFrame, KFOLDS:int=5) -> List[ValidationFold]:
    shuffled_data = df.sample(frac=1)
    shuffled_data.reset_index(inplace=True)

    folds = []
    start_idx, step_length = 0, math.floor(shuffled_data.shape[0] / KFOLDS)
    for kfold_nb in range(KFOLDS):
        end_idx = step_length * (kfold_nb + 1)
        folds.append(shuffled_data[start_idx:end_idx])
        start_idx = end_idx

    validation_folds = []
    for val_fold_nb in range(KFOLDS):
        train = pd.concat([fold for nb_fold, fold in enumerate(folds) if nb_fold != val_fold_nb], axis=0)
        validation_folds.append(ValidationFold(train, folds[val_fold_nb], f'val_fold_{val_fold_nb}'))

    return validation_folds

def run_hyperparameter_search_regularized_model(
        formula: str,
        data: pd.DataFrame,
        exposure: pd.Series,
        family: Union[
            statsmodels.genmod.families.family.Poisson,
            statsmodels.genmod.families.family.Gamma,
        ],
        alpha_range: Iterable[float],
        L1_wt: float,
        target:str,
        loss_function
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reg_results = pd.DataFrame()
    deviances = defaultdict(list)
    for alpha in tqdm(alpha_range):
        model = fit_regularized_model(formula,
                                      data,
                                      exposure,
                                      family,
                                      alpha=alpha,
                                      L1_wt=L1_wt  # if L1_wt --> lasso
                                      )
        reg_result = pd.DataFrame(model.params, columns=[alpha]).transpose()
        reg_results = pd.concat([reg_results, reg_result])
        y_pred, y_true = model.predict(data), data[target]
        poisson_dev = loss_function(y_true, y_pred)
        deviances['deviance'].append(poisson_dev)
        deviances['alpha'].append(alpha)
        deviances['L1_wt'].append(L1_wt)
    return reg_results, deviances

def fit_regularized_model(
    formula: str,
    data: pd.DataFrame,
    exposure: pd.Series,
    family: Union[
        statsmodels.genmod.families.family.Poisson,
        statsmodels.genmod.families.family.Gamma,
    ],
    alpha: float,
    L1_wt:float
    ) -> statsmodels.genmod.generalized_linear_model.GLMResultsWrapper:
        return smf.glm(
            formula,
            data=data,
            exposure=exposure,
            family=family,
        ).fit_regularized(method="elastic_net", alpha=alpha, L1_wt=L1_wt)


def run_glm_backward_selection(
    DATASET: pd.DataFrame,
    X: pd.DataFrame,
    params: Dict[str, Any],
    target: str,
    has_exposure:bool,
    family: Union[
        statsmodels.genmod.families.family.Poisson,
        statsmodels.genmod.families.family.Gamma,
    ],
    p_value_lvl: float = 0.05,
) -> statsmodels.genmod.generalized_linear_model.GLMResultsWrapper:
    exposure_name = params.get(Constants.EXPOSURE_NAME)

    DATASET_CP, X_CP = DATASET.copy(), X.copy()
    formula = f"{target} ~ " + " + ".join(
        [c for c in X_CP.columns if c != exposure_name]
    )
    exposure = DATASET_CP[exposure_name] if has_exposure else None
    model = smf.glm(
        formula=formula,
        data=DATASET_CP,
        exposure=exposure,
        family=family,
    ).fit()
    max_p_value = model.pvalues.max()
    if max_p_value > p_value_lvl:
        feature_to_remove = model.pvalues.idxmax()
        try:
            X_CP.drop(columns=feature_to_remove, inplace=True)
        except KeyError as e:
            if (
                feature_to_remove == "Intercept"
            ):  # here we do not want to remove the Intercept as this is not a variable from the dataset
                max_p_value = model.pvalues.sort_values(ascending=False).head(2).min()
                if max_p_value > p_value_lvl:
                    feature_to_remove = (
                        model.pvalues.sort_values(ascending=False).head(2).idxmin()
                    )
                    X_CP.drop(columns=feature_to_remove, inplace=True)
                else:
                    return model
            else:
                raise e
        DATASET_CP.drop(columns=feature_to_remove, inplace=True)
        print(
            f"The feature {feature_to_remove} is removed because it exhibited a p-value of {max_p_value}>{p_value_lvl}"
        )
        return run_glm_backward_selection(
            DATASET_CP, X_CP, params, target, has_exposure=has_exposure, family=family, p_value_lvl=p_value_lvl
        )
    else:
        return model


def get_reference_classes(
    df: pd.DataFrame, target: str,
        exposure_name: str=None,
        valid_for_frequency:bool=True,
        valid_for_severity:bool=False,

) -> List[str]:
    valid_for_frequency = not(valid_for_severity)
    valid_for_severity = not(valid_for_frequency)
    if valid_for_frequency:
        df["claim_frequency"] = df[target] / df[exposure_name]
        target = "claim_frequency"
    reference_class_lst = []
    for cat_var in df.select_dtypes(include=["category", "uint8"]):
        avg_target_per_category = pd.DataFrame(
            df.groupby(cat_var)[target].mean()
        ).sort_values(by=[target], ascending=False)
        reference_class = avg_target_per_category.tail(1).last_valid_index()
        print(f"The reference class of {cat_var} is {reference_class}")
        reference_class_lst.append(f"{cat_var}_{reference_class}")
    if valid_for_frequency:
        df.drop(columns=target, inplace=True)
    return reference_class_lst


def bin_numerical_variables(
    df: pd.DataFrame, var_name: str, nb_bin: int = 5
) -> pd.DataFrame:
    labels = [
        str(interval)
        for interval in list(pd.qcut(df[var_name], q=nb_bin).unique().categories)
    ]
    df[f"{var_name}_bin"] = pd.Categorical(
        pd.qcut(df[var_name], q=nb_bin, labels=labels)
    )
    return df


def apply_mean_hot_categorical_encoding(
    df: pd.DataFrame,
    target: str,
    var_name: str,
    nb_bin: int = 3,
    risk_groups: list = None,
) -> Tuple[pd.DataFrame, Dict]:
    avg_frequency = pd.DataFrame(df.groupby(var_name)[target].mean())
    avg_freq_df = pd.DataFrame(pd.qcut(avg_frequency[target], q=nb_bin))
    categories = avg_freq_df[target].unique().categories

    if not risk_groups:
        risk_groups = [f"low", f"medium", f"high"]
    if nb_bin != len(risk_groups):
        raise ValueError(
            f"nb_bin {nb_bin} and lenght risk_groups {len(risk_groups)} should be the same!"
        )

    cond_list = [
        (avg_frequency[target] > category.left)
        & (avg_frequency[target] <= (category.right + 0.001))
        for category in categories
    ]

    avg_frequency[f"{var_name}_risk_group"] = np.select(cond_list, risk_groups)
    risk_group_mapping = avg_frequency[f"{var_name}_risk_group"].to_dict()
    df[f"{var_name}_risk_group"] = pd.Categorical(
        df.district.replace(risk_group_mapping)
    )
    return df, risk_group_mapping


def get_avg_target_per_numerical_bin(
    df: pd.DataFrame, feature: str, target: Union[str, List[str]]
) -> Union[pd.DataFrame, pd.Series]:
    df_cp = df.copy()
    binned_feature = f"{feature}_bin_quantile_based"
    df_cp[binned_feature] = pd.qcut(df[feature], q=10, precision=0, duplicates="drop")
    return df_cp.groupby(binned_feature)[target].mean()


def get_distribution_info_for_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a dataframe with information about the distribution of the categorical variables
    """
    info = defaultdict(list)
    cat_data = df.select_dtypes(include=["object", "category"])
    cols = cat_data.columns
    for col in cols:
        nb_unique_values, dist = get_distribution(df[col])
        info["nb_unique_categories"].append(nb_unique_values), info[
            "distribution"
        ].append(dist)
    return pd.DataFrame(info, index=cols)


def get_distribution(s: pd.Series) -> Tuple[int, Dict[str, str]]:
    """
    Returns the unique number of categories and an ordered format distribution for categorical variables
    """
    dist = s.value_counts(dropna=False, normalize=True).to_dict()
    dist_ordered = dict(sorted(dist.items(), key=lambda item: item[1], reverse=True))
    return len(s.unique()), {k: "{:.2%}".format(v) for k, v in dist_ordered.items()}
