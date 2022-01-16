from typing import Dict, Tuple, List, Union, Any, Iterable
import pandas as pd
from collections import defaultdict
import numpy as np
import statsmodels.genmod.generalized_linear_model

from src.constants import Constants, params
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance

__all__ = [
    "get_distribution",
    "get_distribution_info_for_categorical_variables",
    "get_avg_target_per_numerical_bin",
    "apply_mean_hot_categorical_encoding",
    "bin_numerical_variables",
    "get_reference_classes",
    "run_glm_backward_selection",
    'fit_regularized_model',
    'run_hyperparameter_search_regularized_model'
]

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
    model = smf.glm(
        formula=formula,
        data=DATASET_CP,
        exposure=DATASET_CP[exposure_name],
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
            DATASET_CP, X_CP, params, target, family=family, p_value_lvl=p_value_lvl
        )
    else:
        return model


def get_reference_classes(
    df: pd.DataFrame, target: str, exposure_name: str
) -> List[str]:
    reference_class_lst = []
    df["claim_frequency"] = df[target] / df[exposure_name]
    for cat_var in df.select_dtypes(include=["category", "uint8"]):
        avg_frequency_per_category = pd.DataFrame(
            df.groupby(cat_var)["claim_frequency"].mean()
        ).sort_values(by=["claim_frequency"], ascending=False)
        reference_class = avg_frequency_per_category.tail(1).last_valid_index()
        print(f"The reference class of {cat_var} is {reference_class}")
        reference_class_lst.append(f"{cat_var}_{reference_class}")
    df.drop(columns="claim_frequency", inplace=True)
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
