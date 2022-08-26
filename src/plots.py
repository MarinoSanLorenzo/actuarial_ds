import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.constants import Constants, params, Constants2, params_blog_2
from typing import List, Dict, Any
from src.utils import get_avg_target_per_numerical_bin

__all__ = [
    "plot_univariate_numerical_variables_distribution",
    "plot_univariate_categorical_variables_distribution",
    "plot_avg_target_per_numerical_bin",
    "plt_avg_target_per_category",
    'plot_params_vs_alpha'
]

def plot_params_vs_alpha(reg_results:pd.DataFrame) -> None:
    reg_results.plot()
    plt.xlabel('alpha')
    plt.ylabel('coefficient')
    plt.title('Coefficient by alpha value')

def plt_avg_target_per_category(
    df: pd.DataFrame,
    categorical_variables: List[str],
    params: Dict[str, Any],
    target: str = None,
) -> None:
    nb_claims = params.get(Constants.NB_CLAIMS)
    if not target:
        target = nb_claims
    for feature in categorical_variables:
        avg_target = pd.DataFrame(df.groupby(feature)[target].mean()).sort_values(
            by=[target], ascending=False
        )
        avg_nb_claims = pd.DataFrame(df.groupby(feature)[nb_claims].mean()).sort_values(
            by=[nb_claims], ascending=False
        )
        pd.merge(avg_target, avg_nb_claims, on=feature).plot.bar()
        plt.title(f"Average {target} and {nb_claims} per category of {feature.upper()}")
        plt.ylabel(f"Average {target} and {nb_claims}")
        plt.xlabel(f"{feature.upper()}")
        plt.show()
        plt.close()


def plot_avg_target_per_numerical_bin(
    df: pd.DataFrame,
    numerical_variables: List[str],
    params: Dict[str, Any],
    target: str = None,
) -> None:
    nb_claims, claim_amount, var_to_exclude = (
        params.get(Constants2.NB_CLAIMS),
        params.get(Constants2.CLAIM_AMOUNT),
        params.get(Constants2.VARIABLES_TO_EXCLUDE),
    )
    if not target:
        target = nb_claims
    for num_var in numerical_variables:
        if not num_var in {
            *params.get(Constants2.VARIABLES_TO_EXCLUDE),
            nb_claims,
            claim_amount,
            target,
        }:
            avg_claim_frequency_per_bin = get_avg_target_per_numerical_bin(
                df, num_var, target
            )
            avg_nb_claims_per_bin = get_avg_target_per_numerical_bin(
                df, num_var, nb_claims
            )
            avg_claim_frequency_per_bin.plot(label=target)
            avg_nb_claims_per_bin.plot(label=nb_claims)
            plt.ylabel(f"Average {target} and {nb_claims}")
            plt.xticks(rotation=90)
            plt.title(f"Average {target} and {nb_claims} per bin {num_var}")
            plt.legend()
            plt.show()
            plt.close()


def plot_univariate_categorical_variables_distribution(
    df: pd.DataFrame, categorical_variables: List[str], params: Dict[str, Any]
) -> None:
    for cat_var in categorical_variables:
        if cat_var not in params.get(Constants.VARIABLES_TO_EXCLUDE):
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            dist = df[cat_var].value_counts().head(10) / df.shape[0]
            idx, values = list(dist.index), list(dist.values)
            ax.bar(idx, values)
            plt.title(f"Bin distribution of {cat_var}")
            plt.xlabel(f"Categories of {cat_var}")
            plt.ylabel("Value count")
            plt.xticks(rotation=90)
            plt.show()
            plt.close()


def plot_univariate_numerical_variables_distribution(
    df: pd.DataFrame, numerical_variables: List[str], params: Dict[str, Any]
) -> None:
    for num_var in numerical_variables:
        if num_var not in params.get(Constants2.VARIABLES_TO_EXCLUDE):
            sns.distplot(df[num_var])
            plt.show()
            plt.close()
