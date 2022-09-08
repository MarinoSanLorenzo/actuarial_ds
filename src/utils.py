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
import sklearn
from sklearn.tree import DecisionTreeRegressor
import keras
from keras.models import Sequential
from keras import Input  # for instantiating a keras tensor
from keras.layers import (
    Dense,
    multiply,
    Dropout,
)  # for creating regular densely-connected NN layers.
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_poisson_deviance
from typing import *
from pprint import pprint

from typing import NamedTuple
from collections import namedtuple


__all__ = [
    "get_distribution",
    "get_distribution_info_for_categorical_variables",
    "get_avg_target_per_numerical_bin",
    "apply_mean_hot_categorical_encoding",
    "bin_numerical_variables",
    "get_reference_classes",
    "run_glm_backward_selection",
    "fit_regularized_model",
    "run_hyperparameter_search_regularized_model",
    "get_validation_folds",
    "run_cross_validation",
    "run_hyperopt",
    "get_features_modalities",
    "get_nb_max_combinations",
    "select_random_modality_vector",
    "generate_all_profiles",
    "does_profile_belong_to_ref_class",
    "PurePremiunProfile",
    "get_risk_premium",
    "does_profile_belong_to_ref_class",
    "get_all_pure_premium_profiles",
    "run_hyperopt_for_tree_based_methods",
    "fitting_regression_tree",
    "fit_feed_forward_neural_network",
    "run_optimization_neural_network",
]

ValidationFold = namedtuple("ValidationFold", "train validation name")


def run_optimization_neural_network(
    x_train: np.array,
    y_train: np.array,
    x_val: np.array,
    y_val: np.array,
    x_test: np.array,
    y_test: np.array,
    exp_test: np.array,
    hyperparameter_space: Dict[str, Iterable[Any]],
    n_max_experiments: int = 5,
    max_optimization_time: int = 60,
) -> Tuple[pd.DataFrame, keras.engine.sequential.Sequential]:
    results = defaultdict(list)
    nb_experiment = 0
    start = time.time()
    elapsed_time = time.time() - start
    layer_hyperparameters = [
        "activation",
        "dropout_rate",
        "units",
        "use_bias",
        "kernel_initializer",
    ]
    best_deviance_score, best_model = math.inf, None
    while nb_experiment < n_max_experiments and elapsed_time < max_optimization_time:
        print(f'{"-" * 50} {nb_experiment}th EXPERIMENT {"-" * 50}')
        selected_hyperparams = {}
        for param_name, v in hyperparameter_space.items():
            if all(
                [param_name != layer_param for layer_param in layer_hyperparameters]
            ):
                selected_hyperparams[param_name] = random.choice(v)
        else:
            for n_layer in range(
                selected_hyperparams.get("nb_hidden_layers")
            ):  # randomly set hyperparam for each layer of the NN
                selected_hyperparams[f"layer_param_{n_layer}"] = {
                    k: random.choice(v)
                    for k, v in hyperparameter_space.items()
                    if k in layer_hyperparameters
                }
        pprint(selected_hyperparams)
        model = fit_feed_forward_neural_network(
            x_train, y_train, x_val, y_val, params=selected_hyperparams
        )
        y_pred = model.predict(x_test)
        poisson_dev = mean_poisson_deviance(
            y_test, y_pred[:, 0], sample_weight=exp_test
        )
        results["selected_hyperparams"].append(selected_hyperparams)
        results["poisson_dev"].append(poisson_dev)
        if poisson_dev < best_deviance_score:
            best_deviance_score = poisson_dev
            best_model = model
        nb_experiment += 1
        elapsed_time = time.time() - start
    else:
        results_df = pd.DataFrame.from_dict(results)
        results_df.sort_values(by="poisson_dev", ascending=True, inplace=True)
        results_df.reset_index(inplace=True, drop=True)
    return results_df, best_model


def fit_feed_forward_neural_network(
    x_train: np.array,
    y_train: np.array,
    x_val: np.array,
    y_val: np.array,
    params: Dict[str, Any],
) -> keras.engine.sequential.Sequential:
    #     x_train = params.get('x_train')
    #     y_train = params.get('y_train')
    nb_hidden_layers = params.get("nb_hidden_layers")
    optimizer = params.get("optimizer")
    batch_size = params.get("batch_size")
    dropout_rate = params.get("dropout_rate")
    callbacks = params.get("callbacks")

    model = Sequential()
    model.add(Input(shape=(None, x_train.shape[1])))
    for n_layer in range(nb_hidden_layers):
        layer_params = params.get(f"layer_param_{n_layer}")
        dropout_rate = layer_params.get("dropout_rate")
        del layer_params["dropout_rate"]
        model.add(Dense(**layer_params))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="exponential"))
    model.compile(
        optimizer=optimizer,  # default='rmsprop', an algorithm to be used in backpropagation
        loss=tf.keras.losses.poisson,
        # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
        metrics=tf.keras.metrics.Poisson(),
        # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        loss_weights=None,
        # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
        weighted_metrics=None,
        # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
        run_eagerly=None,
        # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
        steps_per_execution=None
        # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
    )
    ##### Step 5 - Fit keras model on the dataset
    model.fit(
        x_train,  # input data
        y_train,  # target data
        batch_size=batch_size,
        # Number of samples per gradient update. If unspecified, batch_size will default to 32.
        epochs=10,
        # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
        verbose="auto",
        # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
        callbacks=callbacks,  # default=None, list of callbacks to apply during training. See tf.keras.callbacks
        validation_split=0.2,
        # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
        validation_data=(x_val, y_val),
        # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
        shuffle=True,
        # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
        class_weight=None,
        # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
        sample_weight=None,
        # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
        initial_epoch=0,
        # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
        steps_per_epoch=None,
        # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
        validation_steps=None,
        # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
        validation_batch_size=None,
        # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
        validation_freq=1,
        # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
        max_queue_size=10,
        # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        workers=1,
        # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
        use_multiprocessing=False,
        # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
    )
    return model


def run_cross_validation_for_tree_based_methods(
    df_train: pd.DataFrame,
    model_fit_func: FunctionType,
    params_models: Dict[str, Iterable],
    loss_function: Union[FunctionType],
    params_to_record: List[str] = None,
    exposure_name: str = params.get(Constants.EXPOSURE_NAME),
    target_name: str = params.get(Constants.CLAIM_FREQUENCY),
) -> Dict[str, List]:
    """
    :param df: Train Dataset on which the data fold will be build
    :param model_fit_func: a custom/user-defined wrapper function fitting a model
    :param params_models: a dictionary with the keys as parameter of the model_fit_func
    :param loss_function: a user-defined function or sklearn.metric to assess the performance on the k-fold
    :param params_to_record: a list containing all the hyperparam name to record
    :return: a Dictionary with hyperparameters values, loss and models
    """

    if not params_to_record:
        params_to_record = list(params_models.keys())
    validation_folds = get_validation_folds(df_train)
    results = defaultdict(list)

    for validation_fold in validation_folds:
        x_train = validation_fold.train
        exp_train, y_train = x_train[exposure_name], x_train[target_name]
        x_train.drop(columns=[exposure_name, target_name], inplace=True)

        model = model_fit_func(params_models, x_train, y_train, exp_train)

        val_data = validation_fold.validation
        val_x = val_data.drop(columns=[exposure_name, target_name])
        y_pred_val, y_true_val, exp_val = (
            model.predict(val_x),
            val_data[target_name],
            val_data[exposure_name],
        )

        if exposure_name:
            val_loss = loss_function(y_true_val, y_pred_val, sample_weight=exp_val)
        else:
            val_loss = loss_function(y_true_val, y_pred_val)

        results["loss"].append(val_loss)
        results["model"].append(model)

        for param_name in params_to_record:
            results[param_name].append(params_models.get(param_name))
    return results


def fitting_regression_tree(
    params_model: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train: Union[pd.Series, pd.DataFrame],
    exp_train: Union[pd.Series, pd.DataFrame],
) -> DecisionTreeRegressor:
    poisson_tree = DecisionTreeRegressor(**params_model)
    poisson_tree.fit(x_train, y_train, sample_weight=exp_train)
    return poisson_tree


def run_hyperopt_for_tree_based_methods(
    df_train: pd.DataFrame,
    hyperparams_space: Dict[str, Iterable],
    model_fit_func: FunctionType,
    loss_function: Union[FunctionType],
    params_to_record: List[str] = None,
    exposure_name: str = params.get(Constants.EXPOSURE_NAME),
    target_name: str = params.get(Constants.CLAIM_FREQUENCY),
    limit_time: int = 100,
    max_iter: int = 100,
    is_debug: bool = False,
) -> Tuple[pd.DataFrame, DecisionTreeRegressor]:
    """
    :param df: Train Dataset on which the data fold will be build
    :param hyperparams_space: dictionarry with the name (that the model_fit_func takes) of the hyperparam and an iterable containing all values to sample from
    :param model_fit_func:  a custom/user-defined wrapper function fitting a model
    :param loss_function:  a user-defined function or sklearn.metric to assess the performance on the k-fold
    :param params_to_record: a list containing all the hyperparam name to record
    :param limit_time: maximum time in seconds to run the hyperopt
    :param max_iter: number of max iterations to run the hyperopt
    :return:
    """
    max_combinations = np.cumprod([len(v) for v in hyperparams_space.values()])[-1]
    tried_hyperparam = set()
    if not max_iter:
        max_iter = max_combinations
    else:
        max_iter = min(max_iter, max_combinations)
    if is_debug:
        print(f"the number of the maximum iterations is {max_iter}")
    nb_iter = 1
    start = time.time()
    duration = time.time() - start
    cv_results = defaultdict(list)
    best_models = None
    best_val_score = math.inf
    while (nb_iter <= max_iter) and (duration < limit_time):
        if is_debug:
            print(f"--------------------------------")
            print(f"{nb_iter} iteration starting...")
            print(f"{duration} time elapsed...")
        random_hyper_param = {
            hyperparam: random.choice(range_)
            for hyperparam, range_ in hyperparams_space.items()
        }
        hyper_space_combi = "-".join(
            [str(f"{p}:{random_hyper_param.get(p)}") for p in params_to_record]
        )
        if (
            hyper_space_combi in tried_hyperparam
        ):  # do not try an aleady tested hyperparam combination
            continue

        cv_result = run_cross_validation_for_tree_based_methods(
            df_train=df_train,
            model_fit_func=model_fit_func,
            params_models=random_hyper_param,
            loss_function=loss_function,
            params_to_record=params_to_record,
            exposure_name=exposure_name,
            target_name=target_name,
        )

        cv_mean_loss = np.mean(cv_result["loss"])
        cv_std_loss = np.std(cv_result["loss"])
        if cv_mean_loss < best_val_score:
            best_models = cv_result["model"]
            best_val_score = cv_mean_loss

        cv_results["losses"].append(cv_result["loss"])
        cv_results["cv_mean_loss"].append(cv_mean_loss)
        cv_results["cv_std_loss"].append(cv_std_loss)
        cv_results["hyperparams"].append(hyper_space_combi)
        for param_to_record in params_to_record:
            cv_results[param_to_record].append(random_hyper_param[param_to_record])

        tried_hyperparam.add(hyper_space_combi)
        nb_iter += 1
        duration = time.time() - start

    # TODO: return the best score based on the error quadratic lower mean loss and lower std deviation
    return pd.DataFrame.from_dict(cv_results), best_models


class PurePremiunProfile(NamedTuple):
    profile_name: str
    features_lst: List[str]
    mean_freq: float
    mean_sev: float
    pure_premium_risk: float
    calc_details: str


def get_all_pure_premium_profiles(
    features_modalities: Dict[str, List[str]],
    exp_params_freq: Dict[str, float],
    exp_params_sev: Dict[str, float],
    reference_class_lst: List[str],
    reference_class_sev_lst: List[str],
    sep_token: str = "---",
) -> Dict[str, PurePremiunProfile]:
    unique_profiles = generate_all_profiles(features_modalities, sep_token=sep_token)
    premium_profiles = defaultdict(list)
    for profile in unique_profiles:
        profile_parsed = profile.split(sep_token)
        pp_freq = get_risk_premium(profile_parsed, exp_params_freq, reference_class_lst)
        pp_sev = get_risk_premium(
            profile_parsed, exp_params_sev, reference_class_sev_lst
        )
        mean_freq, mean_sev = pp_freq.risk_premium, pp_sev.risk_premium
        premium_profiles[profile] = PurePremiunProfile(
            profile_name=profile,
            features_lst=profile_parsed,
            mean_freq=mean_freq,
            mean_sev=mean_sev,
            pure_premium_risk=mean_freq * mean_sev,
            calc_details=f"{pp_freq.formula_with_num_details}*{pp_sev.formula_with_num_details}",
        )
    return premium_profiles


class RiskPremiumCalculationDetail(NamedTuple):
    risk_premium: float
    formula_with_num_details: str


def get_risk_premium(
    profile: List[str], exp_params_risk: Dict[str, float], reference_class: List[str]
) -> float:
    risk_premium = exp_params_risk.get("Intercept")
    formula_with_num_details = f"(exp(Intercept)={round(risk_premium,2)})"

    if does_profile_belong_to_ref_class(profile, reference_class):
        return RiskPremiumCalculationDetail(
            risk_premium=risk_premium, formula_with_num_details=formula_with_num_details
        )
    formula_with_num_details_lst = [formula_with_num_details]
    for modality in profile:
        exp_param_risk = exp_params_risk.get(modality, 1)
        risk_premium *= exp_param_risk
        formula_with_num_details = f"(exp({modality})={round(exp_param_risk, 2)})"
        formula_with_num_details_lst.append(formula_with_num_details)

    formula_with_num_details = "*".join(formula_with_num_details_lst)
    return RiskPremiumCalculationDetail(
        risk_premium=risk_premium, formula_with_num_details=formula_with_num_details
    )


def does_profile_belong_to_ref_class(
    profile: List[str], reference_class_lst: List[str]
) -> bool:
    return all([modality in reference_class_lst for modality in profile])


def generate_all_profiles(
    features_modallities: Dict[str, List[str]], sep_token: str
) -> Set[str]:
    nb_max_combinations = get_nb_max_combinations(features_modallities)
    unique_profiles = set()
    while len(unique_profiles) != nb_max_combinations:
        modality_vector = select_random_modality_vector(features_modallities)
        modality_vector_str = sep_token.join(modality_vector)
        if modality_vector_str not in unique_profiles:
            unique_profiles.add(modality_vector_str)
    return unique_profiles


def select_random_modality_vector(
    features_modallities: Dict[str, List[str]]
) -> List[str]:
    return [
        random.choice(modality_lst) for modality_lst in features_modallities.values()
    ]


def get_nb_max_combinations(features_modallities: Dict[str, List[str]]) -> int:
    nb_max_combinations = 1
    for v in features_modallities.values():
        nb_max_combinations *= len(v)
    return nb_max_combinations


def get_feature_from_modality(feature_modality: str):
    return feature_modality[: re.search("\_", feature_modality).start()]


def get_features_modalities(
    scorecard: pd.DataFrame,
    reference_class_lst: List[str],
    reference_class_sev_lst: List[str],
):
    reference_class_set = set(reference_class_lst) | set(reference_class_sev_lst)
    all_features_modalities = set(set(scorecard.index) | reference_class_set) - set(
        ["Intercept"]
    )
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


def run_hyperopt(
    df: pd.DataFrame,
    hyperparams_space: Dict[str, Iterable],
    model_fit_func: FunctionType,
    loss_function: Union[FunctionType],
    params_to_record: List[str] = None,
    limit_time: int = 100,
    max_iter: int = 100,
    exposure_name: str = "exposure",
    target_name: str = "target",
    is_debug: bool = False,
) -> pd.DataFrame:
    """
    :param df: Train Dataset on which the data fold will be build
    :param hyperparams_space: dictionarry with the name (that the model_fit_func takes) of the hyperparam and an iterable containing all values to sample from
    :param model_fit_func:  a custom/user-defined wrapper function fitting a model
    :param loss_function:  a user-defined function or sklearn.metric to assess the performance on the k-fold
    :param params_to_record: a list containing all the hyperparam name to record
    :param limit_time: maximum time in seconds to run the hyperopt
    :param max_iter: number of max iterations to run the hyperopt
    :return:
    """
    max_combinations = np.cumprod([len(v) for v in hyperparams_space.values()])[-1]
    tried_hyperparam = set()
    if not max_iter:
        max_iter = max_combinations
    else:
        max_iter = min(max_iter, max_combinations)
    if is_debug:
        print(f"the number of the maximum iterations is {max_iter}")
    nb_iter = 1
    start = time.time()
    duration = time.time() - start
    cv_results = defaultdict(list)
    while (nb_iter <= max_iter) and (duration < limit_time):
        if is_debug:
            print(f"--------------------------------")
            print(f"{nb_iter} iteration starting...")
            print(f"{duration} time elapsed...")
        random_hyper_param = {
            hyperparam: random.choice(range_)
            for hyperparam, range_ in hyperparams_space.items()
        }
        hyper_space_combi = "-".join(
            [str(f"{p}:{random_hyper_param.get(p)}") for p in params_to_record]
        )
        if (
            hyper_space_combi in tried_hyperparam
        ):  # do not try an aleady tested hyperparam combination
            continue
        cv_result = run_cross_validation(
            df=df,
            model_fit_func=model_fit_func,
            params_models=random_hyper_param,
            loss_function=loss_function,
            params_to_record=params_to_record,
            exposure_name=exposure_name,
            target_name=target_name,
        )

        cv_results["losses"].append(cv_result["loss"])
        cv_results["cv_mean_loss"].append(np.mean(cv_result["loss"]))
        cv_results["cv_std_loss"].append(np.std(cv_result["loss"]))
        cv_results["hyperparams"].append(hyper_space_combi)
        for param_to_record in params_to_record:
            cv_results[param_to_record].append(random_hyper_param[param_to_record])

        tried_hyperparam.add(hyper_space_combi)
        nb_iter += 1
        duration = time.time() - start

    # TODO: return the best score based on the error quadratic lower mean loss and lower std deviation
    return pd.DataFrame.from_dict(cv_results)


def run_cross_validation(
    df: pd.DataFrame,
    model_fit_func: FunctionType,
    params_models: Dict[str, Iterable],
    loss_function: Union[FunctionType],
    params_to_record: List[str] = None,
    exposure_name: str = "exposure",
    target_name: str = "target",
) -> Dict[str, List]:
    """
    :param df: Train Dataset on which the data fold will be build
    :param model_fit_func: a custom/user-defined wrapper function fitting a model
    :param params_models: a dictionary with the keys as parameter of the model_fit_func
    :param loss_function: a user-defined function or sklearn.metric to assess the performance on the k-fold
    :param params_to_record: a list containing all the hyperparam name to record
    :return: a Dictionary with hyperparameters values, loss and models
    """

    if not params_to_record:
        params_to_record = list(params_models.keys())
    validation_folds = get_validation_folds(df)
    results = defaultdict(list)

    for validation_fold in validation_folds:
        train_data, exposure = (
            validation_fold.train,
            validation_fold.train[exposure_name],
        )
        params_models["data"], params_models["exposure"] = train_data, exposure
        model = model_fit_func(**params_models)
        y_pred, y_true = (
            model.predict(validation_fold.validation),
            validation_fold.validation[target_name],
        )
        loss = loss_function(y_true, y_pred)
        results["loss"].append(loss)
        results["model"].append(model)
        for param_name in params_to_record:
            results[param_name].append(params_models.get(param_name))
    return results


def get_validation_folds(df: pd.DataFrame, KFOLDS: int = 5) -> List[ValidationFold]:
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
        train = pd.concat(
            [fold for nb_fold, fold in enumerate(folds) if nb_fold != val_fold_nb],
            axis=0,
        )
        validation_folds.append(
            ValidationFold(train, folds[val_fold_nb], f"val_fold_{val_fold_nb}")
        )

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
    target: str,
    loss_function,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reg_results = pd.DataFrame()
    deviances = defaultdict(list)
    for alpha in tqdm(alpha_range):
        model = fit_regularized_model(
            formula,
            data,
            exposure,
            family,
            alpha=alpha,
            L1_wt=L1_wt,  # if L1_wt --> lasso
        )
        reg_result = pd.DataFrame(model.params, columns=[alpha]).transpose()
        reg_results = pd.concat([reg_results, reg_result])
        y_pred, y_true = model.predict(data), data[target]
        poisson_dev = loss_function(y_true, y_pred)
        deviances["deviance"].append(poisson_dev)
        deviances["alpha"].append(alpha)
        deviances["L1_wt"].append(L1_wt)
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
    L1_wt: float,
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
    has_exposure: bool,
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
            DATASET_CP,
            X_CP,
            params,
            target,
            has_exposure=has_exposure,
            family=family,
            p_value_lvl=p_value_lvl,
        )
    else:
        return model


def get_reference_classes(
    df: pd.DataFrame,
    target: str,
    exposure_name: str = None,
    valid_for_frequency: bool = True,
    valid_for_severity: bool = False,
) -> List[str]:
    valid_for_frequency = not (valid_for_severity)
    valid_for_severity = not (valid_for_frequency)
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
