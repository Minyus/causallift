from typing import Any, Dict, Type  # NOQA

from easydict import EasyDict
from kedro.utils import load_obj
import numpy as np
import pandas as pd
import sklearn  # NOQA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_cols_features(
    df,
    non_feature_cols=[
        "Treatment",
        "Outcome",
        "TransformedOutcome",
        "Propensity",
        "Recommendation",
    ],
):
    return [column for column in df.columns if column not in non_feature_cols]


def concat_train_test(args, train, test):
    r"""
    Concatenate train and test series.
    Use series.xs('train') or series.xs('test') to split
    """
    series = pd.concat(
        [pd.Series(train), pd.Series(test)],
        keys=["train", "test"],
        names=[args.partition_name, args.index_name],
    )
    return series


def concat_train_test_df(args, train, test):
    r"""
    Concatenate train and test data frames.
    Use df.xs('train') or df.xs('test') to split.
    """
    df = pd.concat(
        [train, test],
        keys=["train", "test"],
        names=[args.partition_name, args.index_name],
    )
    return df


def len_t(df, treatment=1.0, col_treatment="Treatment"):
    return df.query("{}=={}".format(col_treatment, treatment)).shape[0]


def len_o(df, outcome=1.0, col_outcome="Outcome"):
    return df.query("{}=={}".format(col_outcome, outcome)).shape[0]


def len_to(
    df, treatment=1.0, outcome=1.0, col_treatment="Treatment", col_outcome="Outcome"
):
    len_ = df.query(
        "{}=={} & {}=={}".format(col_treatment, treatment, col_outcome, outcome)
    ).shape[0]
    return len_


def treatment_fraction_(df, col_treatment="Treatment"):
    return len_t(df, col_treatment=col_treatment) / len(df)


def treatment_fractions_(
    args,  # type: Dict[str, Any]
    df,  # type: Type[pd.DataFrame]
):
    # type: (...) -> Type[EasyDict]
    col_treatment = args.col_treatment
    treatment_fractions = {
        "train": treatment_fraction_(df.xs("train"), col_treatment=col_treatment),
        "test": treatment_fraction_(df.xs("test"), col_treatment=col_treatment),
    }
    return EasyDict(treatment_fractions)


def outcome_fraction_(df, col_outcome="Outcome"):
    return len_o(df, col_outcome=col_outcome) / len(df)


def overall_uplift_gain_(
    df, treatment=1.0, outcome=1.0, col_treatment="Treatment", col_outcome="Outcome"
):
    overall_uplift_gain = (
        len_to(df, col_treatment=col_treatment, col_outcome=col_outcome)
        / len_t(df, col_treatment=col_treatment)
    ) - (
        len_to(df, 0, 1, col_treatment=col_treatment, col_outcome=col_outcome)
        / len_t(df, 0, col_treatment=col_treatment)
    )
    return overall_uplift_gain


def gain_tuple(df_, r_):
    treatment_fraction = treatment_fraction_(df_)
    outcome_fraction = outcome_fraction_(df_)
    overall_uplift_gain = overall_uplift_gain_(df_)

    cgain = np.interp(treatment_fraction, r_.cgains_x, r_.cgains_y)
    cgain_base = overall_uplift_gain * treatment_fraction
    cgain_factor = cgain / cgain_base

    return (
        treatment_fraction,
        outcome_fraction,
        overall_uplift_gain,
        cgain,
        cgain_base,
        cgain_factor,
        r_.Q_cgains,
        r_.q1_cgains,
        r_.q2_cgains,
    )


def score_df(y_train, y_test, y_pred_train, y_pred_test, average="binary"):
    if len(y_train) != len(y_pred_train):
        raise Exception("Lengths of true and predicted for train do not match.")
    if len(y_pred_test) != len(y_pred_test):
        raise Exception("Lengths of true and predicted for test do not match.")
    num_classes = pd.Series(y_train).nunique()
    score_2darray = [
        [
            len(y_),
            pd.Series(y_).nunique(),
            accuracy_score(y_, y_pred_),
            precision_score(y_, y_pred_, average=average),
            recall_score(y_, y_pred_, average=average),
            f1_score(y_, y_pred_, average=average),
        ]
        + (
            [
                roc_auc_score(y_, y_pred_),
                pd.Series(y_).mean(),
                pd.Series(y_pred_).mean(),
            ]
            if num_classes == 2
            else []
        )
        for (y_, y_pred_) in [(y_train, y_pred_train), (y_test, y_pred_test)]
    ]
    score_df = pd.DataFrame(
        score_2darray,
        index=["train", "test"],
        columns=["# samples", "# classes", "accuracy", "precision", "recall", "f1"]
        + (
            ["roc_auc", "observed conversion rate", "predicted conversion rate"]
            if num_classes == 2
            else []
        ),
    )
    return score_df


def conf_mat_df(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    num_class = len(conf_mat)
    true_labels = ["True_{}".format(i) for i in range(num_class)]
    pred_labels = ["Pred_{}".format(i) for i in range(num_class)]
    conf_mat_df = pd.DataFrame(conf_mat, index=true_labels, columns=pred_labels)
    return conf_mat_df


def bundle_train_and_test_data(args, train_df, test_df):
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert set(train_df.columns) == set(test_df.columns)
    assert all([isinstance(col_name, str) for col_name in train_df.columns])

    index_name = args.index_name

    if index_name is not None:
        train_df = train_df.reset_index(drop=True).copy()
        train_df.index.name = index_name
        test_df = test_df.reset_index(drop=True).copy()
        test_df.index.name = index_name
    else:
        assert train_df.index.name == test_df.index.name

    df = concat_train_test_df(args, train_df, test_df)
    return df


def impute_cols_features(args, df):
    non_feature_cols = [
        args.col_treatment,
        args.col_outcome,
        args.col_propensity,
        args.col_cate,
        args.col_recommendation,
    ]

    args.cols_features = args.cols_features or get_cols_features(
        df, non_feature_cols=non_feature_cols
    )
    return args


def compute_cate(proba_treated, proba_untreated):
    cate_estimated = proba_treated - proba_untreated
    return cate_estimated


def add_cate_to_df(args, df, cate_estimated, proba_treated, proba_untreated):
    df.loc[:, args.col_proba_if_treated] = proba_treated
    df.loc[:, args.col_proba_if_untreated] = proba_untreated
    df.loc[:, args.col_cate] = cate_estimated.values
    return df


def recommend_by_cate(args, df, treatment_fractions):
    cate_series = df[args.col_cate]

    def recommendation(cate_series, treatment_fraction):
        rank_series = cate_series.rank(method="first", ascending=False, pct=True)
        r = np.where(rank_series <= treatment_fraction, 1.0, 0.0)
        return r

    recommendation_train = recommendation(
        cate_series.xs("train"), treatment_fractions.train
    )
    recommendation_test = recommendation(
        cate_series.xs("test"), treatment_fractions.test
    )

    df.loc[:, args.col_recommendation] = concat_train_test(
        args, recommendation_train, recommendation_test
    )

    return df


def estimate_effect(sim_treated_df, sim_untreated_df):
    estimated_effect_df = pd.DataFrame()

    estimated_effect_df["# samples"] = (
        sim_treated_df["# samples chosen without uplift model"]
        + sim_untreated_df["# samples chosen without uplift model"]
    )

    ## Original (without uplift model)

    estimated_effect_df["observed conversion rate without uplift model"] = (
        sim_treated_df["# samples chosen without uplift model"]
        * sim_treated_df["observed conversion rate without uplift model"]
        + sim_untreated_df["# samples chosen without uplift model"]
        * sim_untreated_df["observed conversion rate without uplift model"]
    ) / (
        sim_treated_df["# samples chosen without uplift model"]
        + sim_untreated_df["# samples chosen without uplift model"]
    )

    ## Recommended (with uplift model)

    estimated_effect_df["predicted conversion rate using uplift model"] = (
        sim_treated_df["# samples recommended by uplift model"]
        * sim_treated_df["predicted conversion rate using uplift model"]
        + sim_untreated_df["# samples recommended by uplift model"]
        * sim_untreated_df["predicted conversion rate using uplift model"]
    ) / (
        sim_treated_df["# samples recommended by uplift model"]
        + sim_untreated_df["# samples recommended by uplift model"]
    )

    estimated_effect_df["predicted improvement rate"] = (
        estimated_effect_df["predicted conversion rate using uplift model"]
        / estimated_effect_df["observed conversion rate without uplift model"]
    )

    return estimated_effect_df


def initialize_model(
    args,  # type: Type[EasyDict]
    model_key="uplift_model_params",  # type: str
    default_estimator="sklearn.linear_model.LogisticRegression",  # type: str
):
    # type: (...) -> Type[sklearn.base.BaseEstimator]

    if not isinstance(args[model_key], dict):
        model = args[model_key]
        return model

    model_params = args[model_key].copy()
    if not model_params.get("estimator"):
        model_params["estimator"] = default_estimator
    estimator_str = model_params.pop("estimator")
    estimator_obj = load_obj(estimator_str)

    const_params = (
        (model_params.pop("const_params") or dict())
        if "const_params" in model_params
        else dict()
    )

    if not model_params.get("search_cv"):
        const_params.update(model_params)
        model = estimator_obj(**const_params)
        return model

    search_cv_str = model_params.pop("search_cv")
    search_cv_obj = load_obj(search_cv_str)
    model_params["estimator"] = estimator_obj(**const_params)
    model = search_cv_obj(**model_params)
    return model


# class CausalLiftParamError(Exception):
