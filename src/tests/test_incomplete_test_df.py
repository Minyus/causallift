from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from causallift import CausalLift, generate_data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_empty_test_df():

    seed = 0

    df = generate_data(
        N=1000,
        n_features=3,
        beta=[0, -2, 3, -5],  # Effect of [intercept and features] on outcome
        error_std=0.1,
        tau=[1, -5, -5, 10],  # Effect of [intercept and features] on treated outcome
        tau_std=0.1,
        discrete_outcome=True,
        seed=seed,
        feature_effect=0,  # Effect of beta on treated outxome
        propensity_coef=[
            0,
            -1,
            1,
            -1,
        ],  # Effect of [intercept and features] on propensity log-odds for treatment
        index_name="index",
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["Treatment"]
    )

    test_df = None

    cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=3)
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


def test_test_df_without_outcome():

    seed = 0

    df = generate_data(
        N=1000,
        n_features=3,
        beta=[0, -2, 3, -5],  # Effect of [intercept and features] on outcome
        error_std=0.1,
        tau=[1, -5, -5, 10],  # Effect of [intercept and features] on treated outcome
        tau_std=0.1,
        discrete_outcome=True,
        seed=seed,
        feature_effect=0,  # Effect of beta on treated outxome
        propensity_coef=[
            0,
            -1,
            1,
            -1,
        ],  # Effect of [intercept and features] on propensity log-odds for treatment
        index_name="index",
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["Treatment"]
    )

    test_df = test_df.drop(columns=["Outcome"])

    cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=3)
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


def test_single_row_test_df():

    seed = 0

    df = generate_data(
        N=1000,
        n_features=3,
        beta=[0, -2, 3, -5],  # Effect of [intercept and features] on outcome
        error_std=0.1,
        tau=[1, -5, -5, 10],  # Effect of [intercept and features] on treated outcome
        tau_std=0.1,
        discrete_outcome=True,
        seed=seed,
        feature_effect=0,  # Effect of beta on treated outxome
        propensity_coef=[
            0,
            -1,
            1,
            -1,
        ],  # Effect of [intercept and features] on propensity log-odds for treatment
        index_name="index",
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["Treatment"]
    )

    test_df = test_df.head(1)

    cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=3)
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


if __name__ == "__main__":
    # test_empty_test_df()
    test_test_df_without_outcome()
    test_single_row_test_df()
